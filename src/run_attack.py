import torch
from core import iso
from training_utils import scoring_helpers,evaluation, train, parser, save, log_helpers, build
import evaluate as hf_evaluate
import uuid
from training_utils import tensor_board_logging #TODO
from pathlib import Path





def main(args):
    # --- setup ---------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger,start_epoch = build.create_logger(args=args)  # keeps your existing logger setup
    logger.info("Using device: %s", device)
    
    # Build psychoacoustic tools
    #Builds a 2D interpolator:  penalty_weight=interp_phon(phon, freq) , weight in [0,1].
    interp = iso.build_weight_interpolator()#interpolator for iso226 curve weighting. if you sampled a 2D map weights[phon, freq] from ISO-226 RegularGridInterpolator lets you ask:“What’s the perceptual penalty weight at phon=63.2 and freq=912 Hz?
    spl_thresh = build.init_phon_threshold_tensor(args=args)#for projecting according to max phon. creates a max phon allowed limit
    
    #data,model,wer metricc
    train_loader, eval_loader, test_loader, audio_len = build.create_data_loaders(args=args)
    model, processor = build.load_model(args)
    wer_metric = hf_evaluate.load(path="wer", experiment_id=str(uuid.uuid4()))

    # Initialize perturbation
    p = build.init_perturbation(
        spl_thresh=spl_thresh,
        args=args,
        length=audio_len,
        interp=interp,
        first_batch_data=None,  # <- prefer handling inside build.init_perturbation????
    )
    #opt, sched and link it to the perturbation 
    optimizer, scheduler = build.create_optimizer(args=args, p=p)

    # Tracking results
    train_ctc_hist = []
    train_wer_hist = []
    eval_clean_ctc_hist = []
    eval_clean_wer_hist = []
    eval_pert_ctc_hist = []
    eval_pert_wer_hist = []

    #tracking
    best_epoch = -1
    no_improve_epochs = 0
    best_eval_score = float("inf") if args.attack_mode == "targeted" else float("-inf")


    # Paths
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    pert_path = save_dir / "perturbation.pt"

    try:
        # --- training loop ---------------------------------------------------
        for epoch in range(start_epoch, args.num_epochs):

            # Train perturbation
            p, tr_ctc, tr_wer = train.train_epoch(
                args=args,
                train_data_loader=train_loader,
                p=p,
                model=model,
                epoch=epoch,
                processor=processor,
                optimizer=optimizer,
                interp=interp,
                wer_metric=wer_metric,
                spl_thresh=spl_thresh,
            )
            train_ctc_hist.append(tr_ctc)
            train_wer_hist.append(tr_wer)

            # Evaluate clean recordiongs
            clean = evaluation.evaluate(
                args=args,
                data_loader=eval_loader,
                model=model,
                processor=processor,
                wer_metric=wer_metric,
                p=0,
                perturbed=False,
                epoch=epoch,
            )
            eval_clean_ctc_hist.append(clean.ctc)
            eval_clean_wer_hist.append(clean.wer)

            # Evaluate perturbed records
            pert = evaluation.evaluate(
                args=args,
                data_loader=eval_loader,
                model=model,
                processor=processor,
                wer_metric=wer_metric,
                p=p,
                perturbed=True,
                epoch=epoch,
            )
            eval_pert_ctc_hist.append(pert.ctc)
            eval_pert_wer_hist.append(pert.wer)

            # Epoch logging, try to minimicze the prints so its clearer plz! #TODO
            log_helpers.log_epoch_metrics(
                epoch,
                args.num_epochs,
                train_ctc=tr_ctc,
                eval_ctc_clean=clean.ctc,
                eval_ctc_perturbed=pert.ctc,
                train_wer=tr_wer,
                eval_wer_clean=clean.wer,
                eval_wer_perturbed=pert.wer,
            )

            # Persist plots 
            save.save_loss_plot(
                train_scores={"ctc": train_ctc_hist, "wer": train_wer_hist},
                eval_scores_clean={"ctc": eval_clean_ctc_hist, "wer": eval_clean_wer_hist},
                eval_scores_perturbed={"ctc": eval_pert_ctc_hist, "wer": eval_pert_wer_hist},
                save_dir=str(save_dir),
                norm_type=args.norm_type,
            )
            
            #track best
            best_train = scoring_helpers.Scores(
                ctc=scoring_helpers._best_agg(train_ctc_hist, args.attack_mode),
                wer=scoring_helpers._best_agg(train_wer_hist, args.attack_mode),
            )
            best_eval_pert = scoring_helpers.Scores(
                ctc=scoring_helpers._best_agg(eval_pert_ctc_hist, args.attack_mode),
                wer=scoring_helpers._best_agg(eval_pert_wer_hist, args.attack_mode),
            )
            #save res' to json file
            save.save_json_results(
                save_dir=str(save_dir),
                norm_type=args.norm_type,
                attack_size=args.attack_size_string,
                epoch=epoch,
                finished_training=False,
                eval_score_clean={"ctc": clean.ctc, "wer": clean.wer},
                eval_score_perturbed={"ctc": best_eval_pert.ctc, "wer": best_eval_pert.wer},
                train_score={"ctc": best_train.ctc, "wer": best_train.wer},
            )

            # Early stopping on the right signal
            #save if BEST pert yet
            current_metric = pert.wer if args.attack_mode == "targeted" else pert.ctc
            if scoring_helpers._is_better(current_metric, best_eval_score, args.attack_mode):
                no_improve_epochs = 0
                best_eval_score = current_metric
                best_epoch = epoch
                #save
                save.save_pert(p=p, path=str(pert_path))
                save.save_by_epoch(
                    args=args,
                    p=p,
                    test_data_loader=test_loader,
                    model=model,
                    processor=processor,
                    epoch_num=epoch,
                )
            else:
                no_improve_epochs += 1
            
            #use scheduler and step if exists
            if scheduler:
                scheduler.step()
                # Show actual LR(s)
                try:
                    lrs = [f"{lr:.6f}" for lr in scheduler.get_last_lr()]
                    logger.info("[Epoch %d] new LR(s): %s", epoch, ", ".join(lrs))
                except Exception:
                    for i, g in enumerate(optimizer.param_groups):
                        logger.info("[Epoch %d] LR[group %d]: %.6f", epoch, i, g["lr"])
            
            #early stop
            if no_improve_epochs >= args.early_stopping:
                logger.info("No improvements in %d epochs. Stopping early.", no_improve_epochs)
                break

        # --- finalize --------------------------------------------------------
        # Load best perturbation and evaluate on test set
        p = torch.load(pert_path, map_location=device, weights_only=True).to(device)

        finished_training = True
        
        #perturbed test (applyiong p)
        pert_test = evaluation.evaluate(
            args=args,
            data_loader=test_loader,
            model=model,
            processor=processor,
            wer_metric=wer_metric,
            p=p,
            perturbed=True,
            epoch=-1,
        )
        #clean test (no p)
        clean_test = evaluation.evaluate(
            args=args,
            data_loader=test_loader,
            model=model,
            processor=processor,
            wer_metric=wer_metric,
            p=0,#just add zero to avoid if statement later?
            perturbed=False,
            epoch=-1,
        )

        # Final plots & report
        save.save_loss_plot(
            train_scores={"ctc": train_ctc_hist, "wer": train_wer_hist},
            eval_scores_clean={"ctc": eval_clean_ctc_hist, "wer": eval_clean_wer_hist},
            eval_scores_perturbed={"ctc": eval_pert_ctc_hist, "wer": eval_pert_wer_hist},
            save_dir=str(save_dir),
            norm_type=args.norm_type,
            clean_test_loss={"ctc": clean_test.ctc, "wer": clean_test.wer},
            perturbed_test_loss={"ctc": pert_test.ctc, "wer": pert_test.wer},
        )

        # Persist final JSON
        best_train =scoring_helpers.Scores(
            ctc=scoring_helpers._best_agg(train_ctc_hist, args.attack_mode),
            wer=scoring_helpers._best_agg(train_wer_hist, args.attack_mode),
        )
        #update the final json file wiuth the relevant info. try to make it shorter plz
        save.save_json_results(
            save_dir=str(save_dir),
            epoch=best_epoch,
            finished_training=True,
            norm_type=args.norm_type,
            attack_size=args.attack_size_string,
            best_train_score={"ctc": best_train.ctc, "wer": best_train.wer},
            eval_score_clean={"ctc": clean_test.ctc, "wer": clean_test.wer},
            eval_score_perturbed={"ctc": pert_test.ctc, "wer": pert_test.wer},
            final_test_clean={"ctc": clean_test.ctc, "wer": clean_test.wer},
            final_test_perturbed={"ctc": pert_test.ctc, "wer": pert_test.wer},
            best_epoch=best_epoch,
        )

        # TODO TensorBoard summary. tracking The long term scoring results
        # tensor_board_logging.log_experiment_to_tensorboard(
        #     save_dir=args.tensorboard_logger,
        #     norm_type=args.norm_type,
        #     lr=args.lr,
        #     step_size=args.step_size,
        # )

        # Console summary if needed
        log_helpers.log_summary_metrics(
            args=args,
            clean_ctc_test=clean_test.ctc,
            clean_wer_test=clean_test.wer,
            pert_ctc_test=pert_test.ctc,
            pert_wer_test=pert_test.wer,
            best_epoch=best_epoch,
        )
        # if reached here then was able to finish trianing, ret 0
        return 0

    except Exception as e:
        logger.exception("Run failed with an exception: %s", e)
        # still attempt to write a minimal failure report
        try:
            save.save_json_results(
                save_dir=str(save_dir),
                epoch=-1,
                finished_training=False,
                norm_type=args.norm_type,
                attack_size=args.attack_size_string,
                error=str(e),
            )
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    train_args = parser.create_arg_parser().parse_args()
    exit(code=main(args=train_args))