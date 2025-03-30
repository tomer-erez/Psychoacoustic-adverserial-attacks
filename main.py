import torch
import os
import iso
import save
import parser
import build
import eval as eval_
import train
import time
if __name__ == '__main__':
    args = parser.create_arg_parser().parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'using device: {args.device}')
    logger= build.create_logger(args=args)

    interp = iso.build_weight_interpolator()
    freqs_all, phons_all, spl_matrix = iso.compute_iso226_weight_matrix()
    weights_matrix = iso.perceptual_weight(spl_matrix)


    train_data_loader, eval_data_loader,test_data_loader,audio_length = build.create_data_loaders(args=args)
    model,processor = build.load_model(args)
    p,_,_ = build.init_perturbation(args, length=audio_length)


    optimizer= build.create_optimizer(args=args, p=p)
    train_scores = []
    eval_scores_clean = []
    eval_scores_perturbed = []

    no_improve_epochs=0
    best_eval_score_perturbed = -9999

    for epoch in range(args.num_epochs):

        p, train_epoch_score = train.train_epoch(
            args=args, train_data_loader=train_data_loader, p=p,
            model=model,
            epoch=epoch, logger=logger, processor=processor,optimizer=optimizer,
            weights=weights_matrix
        )
        train_scores.append(train_epoch_score)

        eval_epoch_score_clean = eval_.evaluate(
            args=args, eval_data_loader=eval_data_loader, p=0,
            model=model, logger=logger,
            perturbed=False, epoch_number=epoch, processor=processor
        )
        eval_scores_clean.append(eval_epoch_score_clean)

        eval_epoch_score_perturbed = eval_.evaluate(
            args=args, eval_data_loader=eval_data_loader, p=p,
            model=model, logger=logger,
            perturbed=True, epoch_number=epoch, processor=processor
        )
        eval_scores_perturbed.append(eval_epoch_score_perturbed)

        a=f"epoch {epoch}/{args.num_epochs},\teval score clean {eval_epoch_score_clean:.3f},\teval score perturbed {eval_epoch_score_perturbed:.3f},\ttrain score {train_epoch_score:.3f}"
        print(a)
        logger.info(a)

        if eval_epoch_score_perturbed>best_eval_score_perturbed:#update
            no_improve_epochs,best_eval_score_perturbed=0,eval_epoch_score_perturbed
            torch.save(p.detach().cpu(), os.path.join(args.save_dir, f"perturbation_epoch.pt"))
        else:
            no_improve_epochs=no_improve_epochs+1

        if no_improve_epochs>args.early_stopping:#early stopping
            logger.info(f'no improvements in {no_improve_epochs} epochs, stopping training and testing on the test set')
            time.sleep(5)
            break
        save.save_by_epoch(args=args, p=p, test_data_loader=test_data_loader, model=model, processor= processor, epoch_num=epoch)
        save.save_loss_plot(train_scores=train_scores, eval_scores_perturbed=eval_scores_perturbed, eval_scores_clean=eval_scores_clean, save_dir=args.save_dir, norm_type=args.norm_type)

        save.save_json_results(
            save_dir=args.save_dir,norm_type=args.norm_type,attack_size=args.attack_size_string,
            epoch=epoch,train_score=train_epoch_score,eval_score_clean=eval_epoch_score_clean,
            eval_score_perturbed=eval_epoch_score_perturbed
        )

    perturbed_test_score = eval_.evaluate( #eval perturbation
        args=args, eval_data_loader=test_data_loader, p=p,
        model=model, logger=logger,
        perturbed=True, epoch_number=-1, processor=processor
    )
    clean_test_score = eval_.evaluate( #eval clean model
        args=args, eval_data_loader=test_data_loader, p=0,
        model=model, logger=logger,
        perturbed=False, epoch_number=-1, processor=processor
    )
    save.save_loss_plot(
        train_scores=train_scores,eval_scores_clean=eval_scores_clean,
        eval_scores_perturbed=eval_scores_perturbed,save_dir=args.save_dir,
        norm_type=args.norm_type,clean_test_loss=clean_test_score,
        perturbed_test_loss=perturbed_test_score
    )
    save.save_json_results(
        save_dir=args.save_dir,norm_type=args.norm_type,
        attack_size=args.attack_size_string,train_score=train_scores[-1],
        best_train_score=max(train_scores),best_eval_score=best_eval_score_perturbed,
        eval_score_clean=eval_scores_clean[-1],eval_score_perturbed=eval_scores_perturbed[-1],
        final_test_clean=clean_test_score,final_test_perturbed=perturbed_test_score
    )

    x=(f"\nSummary:\n"
                f"Perturbation norm type: {args.norm_type}\n"
                f"Perturbation size: {args.attack_size_string}\n"
                f"best Train score: {max(train_scores):.3f}\t last train score: {train_scores[-1]:.3f}\n"
                f"best Eval score: {best_eval_score_perturbed:.3f}\t last eval score: {eval_scores_perturbed[-1]:.3f}\n"
                f"Perturbed Test score: {perturbed_test_score:.3f}\n"
                f"Clean Test score: {clean_test_score:.3f}\n"
                f"Perturbation efficiency: {(perturbed_test_score/clean_test_score):.3f}\n")
    print(x)
    logger.info(x)