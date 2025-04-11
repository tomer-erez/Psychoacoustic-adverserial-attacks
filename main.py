import torch
import os
from core import iso
from training_utils import evaluation, train, parser, save, log_helpers, build
import evaluate as hf_evaluate
import uuid
from training_utils import tensor_board_logging


def scores(ctc, wer):
    return {"ctc": ctc, "wer": wer}


def main(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    start_epoch=build.create_logger(args=args)

    interp = iso.build_weight_interpolator()
    spl_thresh=build.init_phon_threshold_tensor(args=args)
    train_data_loader, eval_data_loader, test_data_loader, audio_length = build.create_data_loaders(args=args)
    model, processor = build.load_model(args)
    p = build.init_perturbation(spl_thresh=spl_thresh,args=args, length=audio_length, interp=interp, first_batch_data=next(iter(train_data_loader))[0])
    optimizer,scheduler = build.create_optimizer(args=args, p=p)
    wer_metric = hf_evaluate.load(path="wer",experiment_id=str(uuid.uuid4()))

    train_scores = {"ctc": [], "wer": []}
    eval_scores_clean = {"ctc": [], "wer": []}
    eval_scores_perturbed = {"ctc": [], "wer": []}

    no_improve_epochs = 0
    best_eval_score_perturbed = float('inf') if args.attack_mode == "targeted" else float('-inf')
    best_epoch=0
    finished_training =False
    for epoch in range(start_epoch,args.num_epochs):
        ############################################# train epoch on perturbation
        p, train_ctc, train_wer = train.train_epoch(
            args=args, train_data_loader=train_data_loader, p=p,
            model=model, epoch=epoch, processor=processor,
            optimizer=optimizer, interp=interp, wer_metric=wer_metric,spl_thresh=spl_thresh,
        )
        train_scores["ctc"].append(train_ctc)
        train_scores["wer"].append(train_wer)
        #############################################

        ############################################# evaluate clean results
        eval_ctc_clean, eval_wer_clean = evaluation.evaluate(
            args=args, eval_data_loader=eval_data_loader, p=0,
            model=model, wer_metric=wer_metric,
            perturbed=False, epoch_number=epoch, processor=processor
        )
        eval_scores_clean["ctc"].append(eval_ctc_clean)
        eval_scores_clean["wer"].append(eval_wer_clean)
        #############################################

        ############################################# evaluate perturbed results
        eval_ctc_perturbed, eval_wer_perturbed = evaluation.evaluate(
            args=args, eval_data_loader=eval_data_loader, p=p,
            model=model, wer_metric=wer_metric,
            perturbed=True, epoch_number=epoch, processor=processor
        )
        eval_scores_perturbed["ctc"].append(eval_ctc_perturbed)
        eval_scores_perturbed["wer"].append(eval_wer_perturbed)
        #############################################


        log_helpers.log_epoch_metrics( # log epoch
            epoch, args.num_epochs,
            train_ctc=train_ctc, eval_ctc_clean=eval_ctc_clean, eval_ctc_perturbed=eval_ctc_perturbed,
            train_wer=train_wer, eval_wer_clean=eval_wer_clean, eval_wer_perturbed=eval_wer_perturbed
        )

        if args.attack_mode == "targeted":
            best_train_ctc = min(train_scores["ctc"])
            best_train_wer = min(train_scores["wer"])
            best_eval_ctc_perturbed = min(eval_scores_perturbed["ctc"])
            best_eval_wer_perturbed = min(eval_scores_perturbed["wer"])
        else:
            best_train_ctc = max(train_scores["ctc"])
            best_train_wer = max(train_scores["wer"])
            best_eval_ctc_perturbed = max(eval_scores_perturbed["ctc"])
            best_eval_wer_perturbed = max(eval_scores_perturbed["wer"])




        save.save_loss_plot(train_scores=train_scores,eval_scores_perturbed=eval_scores_perturbed,eval_scores_clean=eval_scores_clean,
                            save_dir=args.save_dir,norm_type=args.norm_type)

        save.save_json_results(
            save_dir=args.save_dir, norm_type=args.norm_type,attack_size=args.attack_size_string,
            epoch=epoch,finished_training=finished_training,
            eval_score_clean=scores(eval_ctc_clean,eval_wer_clean)
            ,eval_score_perturbed=scores(best_eval_ctc_perturbed,best_eval_wer_perturbed),
            train_score=scores(best_train_ctc,best_train_wer),
        )

        #early stopping/save epoch and continue
        current_metric = eval_wer_perturbed if args.attack_mode == "targeted" else eval_ctc_perturbed
        if (args.attack_mode == "targeted" and current_metric < best_eval_score_perturbed) or \
           (args.attack_mode == "untargeted" and current_metric > best_eval_score_perturbed):
            no_improve_epochs, best_eval_score_perturbed = 0, current_metric
            save.save_pert(p=p,path=os.path.join(args.save_dir, f"perturbation.pt"))
            save.save_by_epoch(args=args, p=p, test_data_loader=test_data_loader, model=model,processor=processor, epoch_num=epoch)
            best_epoch=epoch
        else:
            no_improve_epochs += 1

        if scheduler:
            scheduler.step()
            for param_group in optimizer.param_groups:
                print(f"[Epoch {epoch}] new LR: {param_group['lr']:.6f}")

        if no_improve_epochs == args.early_stopping:
            print(f'No improvements in {no_improve_epochs} epochs. Stopping early.')
            break

    #== == == == == == == == == == 🏁 FINALIZE TRAINING 🏁 == == == == == == == == == ==
    p = torch.load(os.path.join(args.save_dir, "perturbation.pt"), weights_only=True).to(args.device)

    finished_training=True
    # Final test evaluation
    pert_ctc_test, pert_wer_test = evaluation.evaluate( #test the perturbation
        args=args, eval_data_loader=test_data_loader, p=p,
        model=model, wer_metric=wer_metric,
        perturbed=True, epoch_number=-1, processor=processor
    )
    clean_ctc_test, clean_wer_test = evaluation.evaluate( #test on clean data
        args=args, eval_data_loader=test_data_loader, p=0,
        model=model, wer_metric=wer_metric,
        perturbed=False, epoch_number=-1, processor=processor
    )

    save.save_loss_plot( #create plots for the whole training process
        train_scores=train_scores,
        eval_scores_clean=eval_scores_clean,
        eval_scores_perturbed=eval_scores_perturbed,
        save_dir=args.save_dir,
        norm_type=args.norm_type,
        clean_test_loss=scores(clean_ctc_test, clean_wer_test),
        perturbed_test_loss=scores(pert_ctc_test, pert_wer_test)

    )



    save.save_json_results( # save the result to a json file, easier to later parse and loop over all train logs
        save_dir=args.save_dir,epoch=epoch,finished_training=finished_training,
        norm_type=args.norm_type,
        attack_size=args.attack_size_string,
        best_train_score=scores(best_train_ctc, best_train_wer),
        eval_score_clean=scores(clean_ctc_test, clean_wer_test),
        eval_score_perturbed=scores(pert_ctc_test, pert_wer_test),
        final_test_clean=scores(clean_ctc_test, clean_wer_test),
        final_test_perturbed=scores(pert_ctc_test, pert_wer_test),
        best_epoch=best_epoch
    )
    tensor_board_logging.log_experiment_to_tensorboard(
        save_dir=args.tensorboard_logger,
        norm_type=args.norm_type,
        lr=args.lr,
        step_size = args.step_size,
        gamma=args.gamma,
        optimizer_type=args.optimizer_type,
        norm_size=args.attack_size_string,
        dataset=args.dataset,
        num_epochs=epoch + 1,
        finished_training=finished_training,
        perturbed_test_result=scores(pert_ctc_test, pert_wer_test),
        clean_test_result=scores(clean_ctc_test, clean_wer_test),
        attack_mode=args.attack_mode,
        target=args.target if hasattr(args, "target") else None,
        best_epoch=best_epoch
    )


    log_helpers.log_summary_metrics( # log the important stuff from the training process
    args=args,
    clean_ctc_test=clean_ctc_test,
    clean_wer_test=clean_wer_test,
    pert_ctc_test=pert_ctc_test,
    pert_wer_test=pert_wer_test,
        best_epoch=best_epoch
    )


if __name__ == '__main__':
    args = parser.create_arg_parser().parse_args()
    main(args=args)