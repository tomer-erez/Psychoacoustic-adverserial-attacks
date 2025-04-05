import torch
import os
from core import build, iso, evaluation, train, save, parser
import sys
# import evaluate as hf_evaluate


if __name__ == '__main__':
    args = parser.create_arg_parser().parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    build.create_logger(args=args)
    """START USING THE INTERPOLATOR, currently you only lookup to the nearest bin by discretizing"""
    interp = iso.build_weight_interpolator()
    train_data_loader, eval_data_loader,test_data_loader,audio_length = build.create_data_loaders(args=args)
    model,processor = build.load_model(args)
    p=build.init_perturbation(args=args, length=audio_length,interp=interp,first_batch_data=next(iter(train_data_loader))[0])
    optimizer= build.create_optimizer(args=args, p=p)
    train_scores = []
    eval_scores_clean = []
    eval_scores_perturbed = []
    # wer_metric = hf_evaluate.load("wer")  # global or inside function
    print("why not WER")
    no_improve_epochs=0
    best_eval_score_perturbed = float('inf') if args.attack_mode=="targeted" else float('-inf') #if targeted trying to minimize the loss function which asks how often the target word was generated, if untargeted the loss is how wrong the model is which we wanna maximize with the perturbation

    for epoch in range(args.num_epochs):
        p, train_epoch_score = train.train_epoch(
            args=args, train_data_loader=train_data_loader, p=p,
            model=model,
            epoch=epoch, processor=processor,optimizer=optimizer,
            interp=interp
        )
        train_scores.append(train_epoch_score)

        eval_epoch_score_clean = evaluation.evaluate(
            args=args, eval_data_loader=eval_data_loader, p=0,
            model=model,
            perturbed=False, epoch_number=epoch, processor=processor
        )
        eval_scores_clean.append(eval_epoch_score_clean)

        eval_epoch_score_perturbed = evaluation.evaluate(
            args=args, eval_data_loader=eval_data_loader, p=p,
            model=model,
            perturbed=True, epoch_number=epoch, processor=processor
        )
        eval_scores_perturbed.append(eval_epoch_score_perturbed)
        print(f"epoch {epoch}/{args.num_epochs},\teval score clean {eval_epoch_score_clean:.3f},\teval score perturbed {eval_epoch_score_perturbed:.3f},\ttrain score {train_epoch_score:.3f}")

        if (args.attack_mode == "targeted" and eval_epoch_score_perturbed<best_eval_score_perturbed) or (args.attack_mode == "untargeted" and eval_epoch_score_perturbed>best_eval_score_perturbed):
            no_improve_epochs, best_eval_score_perturbed = 0, eval_epoch_score_perturbed
            torch.save(p.detach().cpu(), os.path.join(args.save_dir, f"perturbation.pt"))
        else:
            no_improve_epochs=no_improve_epochs+1

        if no_improve_epochs>args.early_stopping:#early stopping
            print(f'no improvements in {no_improve_epochs} epochs, stopping training and testing on the test set')
            break

        save.save_by_epoch(args=args, p=p, test_data_loader=test_data_loader, model=model, processor= processor, epoch_num=epoch)
        save.save_loss_plot(train_scores=train_scores, eval_scores_perturbed=eval_scores_perturbed, eval_scores_clean=eval_scores_clean, save_dir=args.save_dir, norm_type=args.norm_type)
        save.save_json_results(
            save_dir=args.save_dir,norm_type=args.norm_type,attack_size=args.attack_size_string,
            epoch=epoch,train_score=train_epoch_score,eval_score_clean=eval_epoch_score_clean,
            eval_score_perturbed=eval_epoch_score_perturbed
        )
        sys.stdout.flush()
        sys.stderr.flush()

    perturbed_test_score = evaluation.evaluate( #eval perturbation
        args=args, eval_data_loader=test_data_loader, p=p,
        model=model,
        perturbed=True, epoch_number=-1, processor=processor
    )
    clean_test_score = evaluation.evaluate( #eval clean model
        args=args, eval_data_loader=test_data_loader, p=0,
        model=model,
        perturbed=False, epoch_number=-1, processor=processor
    )
    save.save_loss_plot(
        train_scores=train_scores,eval_scores_clean=eval_scores_clean,
        eval_scores_perturbed=eval_scores_perturbed,save_dir=args.save_dir,
        norm_type=args.norm_type,clean_test_loss=clean_test_score,
        perturbed_test_loss=perturbed_test_score
    )
    bts = max(train_scores) if args.attack_mode =="untargeted" else min(train_scores)
    save.save_json_results(
        save_dir=args.save_dir,norm_type=args.norm_type,
        attack_size=args.attack_size_string,train_score=train_scores[-1],
        best_train_score=max(train_scores),best_eval_score=best_eval_score_perturbed,
        eval_score_clean=eval_scores_clean[-1],eval_score_perturbed=eval_scores_perturbed[-1],
        final_test_clean=clean_test_score,final_test_perturbed=perturbed_test_score
    )


    print(f"\nSummary:\n"
                f"Perturbation norm type: {args.norm_type}\n"
                f"Perturbation size: {args.attack_size_string}\n"
                f"best Train score: {max(train_scores):.3f}\t last train score: {train_scores[-1]:.3f}\n"
                f"best Eval score: {best_eval_score_perturbed:.3f}\t last eval score: {eval_scores_perturbed[-1]:.3f}\n"
                f"Perturbed Test score: {perturbed_test_score:.3f}\n"
                f"Clean Test score: {clean_test_score:.3f}\n"
                f"Perturbation efficiency: {(perturbed_test_score/clean_test_score):.3f}\n")
    sys.stdout.flush()
    sys.stderr.flush()