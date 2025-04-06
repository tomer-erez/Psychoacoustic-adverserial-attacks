import torch
import os
from core import build, iso, evaluation, train, save, parser,log_helpers
import sys
import evaluate as hf_evaluate
import uuid




if __name__ == '__main__':
    args = parser.create_arg_parser().parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    build.create_logger(args=args)

    interp = iso.build_weight_interpolator()
    train_data_loader, eval_data_loader, test_data_loader, audio_length = build.create_data_loaders(args=args)
    model, processor = build.load_model(args)
    p = build.init_perturbation(args=args, length=audio_length, interp=interp, first_batch_data=next(iter(train_data_loader))[0])
    optimizer = build.create_optimizer(args=args, p=p)
    wer_metric = hf_evaluate.load("wer",experiment_id=str(uuid.uuid4()))

    # Updated to store dicts with "ctc" and "wer"
    train_scores = {"ctc": [], "wer": []}
    eval_scores_clean = {"ctc": [], "wer": []}
    eval_scores_perturbed = {"ctc": [], "wer": []}

    no_improve_epochs = 0
    best_eval_score_perturbed = float('inf') if args.attack_mode == "targeted" else float('-inf')

    for epoch in range(args.num_epochs):

        ############################################# train epoch on perturbation
        p, train_ctc, train_wer = train.train_epoch(
            args=args, train_data_loader=train_data_loader, p=p,
            model=model, epoch=epoch, processor=processor,
            optimizer=optimizer, interp=interp, wer_metric=wer_metric
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

        #early stopping/save epoch and continue
        current_metric = eval_wer_perturbed if args.attack_mode == "targeted" else eval_ctc_perturbed
        if (args.attack_mode == "targeted" and current_metric < best_eval_score_perturbed) or \
           (args.attack_mode == "untargeted" and current_metric > best_eval_score_perturbed):
            no_improve_epochs, best_eval_score_perturbed = 0, current_metric
            torch.save(p.detach().cpu(), os.path.join(args.save_dir, f"perturbation.pt"))
        else:
            no_improve_epochs += 1

        if no_improve_epochs > args.early_stopping:
            print(f'No improvements in {no_improve_epochs} epochs. Stopping early.')
            break

        save.save_by_epoch(args=args, p=p, test_data_loader=test_data_loader,
                           model=model, processor=processor, epoch_num=epoch)
        save.save_loss_plot(train_scores=train_scores,
                            eval_scores_perturbed=eval_scores_perturbed,
                            eval_scores_clean=eval_scores_clean,
                            save_dir=args.save_dir,
                            norm_type=args.norm_type)

        save.save_json_results(
            save_dir=args.save_dir, norm_type=args.norm_type,
            attack_size=args.attack_size_string, epoch=epoch,
            train_score={"ctc": train_ctc, "wer": train_wer},
            eval_score_clean={"ctc": eval_ctc_clean, "wer": eval_wer_clean},
            eval_score_perturbed={"ctc": eval_ctc_perturbed, "wer": eval_wer_perturbed}
        )

    # Final test evaluation
    pert_ctc_test, pert_wer_test = evaluation.evaluate(
        args=args, eval_data_loader=test_data_loader, p=p,
        model=model, wer_metric=wer_metric,
        perturbed=True, epoch_number=-1, processor=processor
    )
    clean_ctc_test, clean_wer_test = evaluation.evaluate(
        args=args, eval_data_loader=test_data_loader, p=0,
        model=model, wer_metric=wer_metric,
        perturbed=False, epoch_number=-1, processor=processor
    )

    save.save_loss_plot(
        train_scores=train_scores,
        eval_scores_clean=eval_scores_clean,
        eval_scores_perturbed=eval_scores_perturbed,
        save_dir=args.save_dir,
        norm_type=args.norm_type,
        clean_test_loss={"ctc": clean_ctc_test, "wer": clean_wer_test},
        perturbed_test_loss={"ctc": pert_ctc_test, "wer": pert_wer_test}
    )

    if args.attack_mode == "targeted":
        best_train_ctc = min(train_scores["ctc"])
        best_train_wer = min(train_scores["wer"])
    else:
        best_train_ctc = max(train_scores["ctc"])
        best_train_wer = max(train_scores["wer"])

    save.save_json_results(
        save_dir=args.save_dir,
        norm_type=args.norm_type,
        attack_size=args.attack_size_string,
        train_score={"ctc": train_scores["ctc"][-1], "wer": train_scores["wer"][-1]},
        best_train_score={"ctc": best_train_ctc, "wer": best_train_wer},
        eval_score_clean={"ctc": clean_ctc_test, "wer": clean_wer_test},
        eval_score_perturbed={"ctc": pert_ctc_test, "wer": pert_wer_test},
        final_test_clean={"ctc": clean_ctc_test, "wer": clean_wer_test},
        final_test_perturbed={"ctc": pert_ctc_test, "wer": pert_wer_test}
    )

    log_helpers.log_summary_metrics(
    args=args,
    best_train_ctc=best_train_ctc,
    best_train_wer=best_train_wer,
    train_scores=train_scores,
    clean_ctc_test=clean_ctc_test,
    clean_wer_test=clean_wer_test,
    pert_ctc_test=pert_ctc_test,
    pert_wer_test=pert_wer_test
    )
