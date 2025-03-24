import build
import eval as eval_
import parser
import train
import save
import torch


if __name__ == '__main__':
    args = parser.create_arg_parser().parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger=build.create_logger(args=args)
    logger.info("hello world")
    train_data_loader, eval_data_loader,test_data_loader,audio_length = build.create_data_loaders(args=args,logger=logger)
    model,processor = build.load_model(args)
    p = build.init_perturbation(args,length=audio_length)
    optimizer=build.create_optimizer(args=args,p=p)
    train_scores = []
    eval_scores = []
    no_improve_epochs=0
    best_eval_score = -9999

    for epoch in range(args.num_epochs):

        p, train_epoch_score = train.train_epoch(
            args=args, train_data_loader=train_data_loader, p=p,
            model=model,
            epoch=epoch, logger=logger, processor=processor,optimizer=optimizer
        )
        train_scores.append(train_epoch_score)

        eval_epoch_score = eval_.evaluate(
            args=args, eval_data_loader=eval_data_loader, p=p,
            model=model, logger=logger,
            perturbed=True, epoch_number=epoch, processor=processor
        )
        eval_scores.append(eval_epoch_score)
        print(f"epoch {epoch}, eval score {eval_epoch_score:.3f}, train score {train_epoch_score:.3f}")

        if eval_epoch_score>best_eval_score:#update
            no_improve_epochs,best_eval_score=0,eval_epoch_score
        else:
            no_improve_epochs=no_improve_epochs+1

        if no_improve_epochs>args.early_stopping:#early stopping
            logger.info(f'no improvements in {no_improve_epochs} epochs, stopping training and testing on the test set')
            break

        save.save_by_epoch(args=args, p=p, test_data_loader=test_data_loader, model=model,processor= processor,epoch_num=epoch)
        save.save_loss_plot(train_scores, eval_scores, args.save_dir)

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
    x=(f"Summary:\n"
                f"Perturbation norm type: {args.norm_type}\n"
                f"Perturbation size: {args.attack_size_string}\n"
                f"best Train score: {max(train_scores):.3f}\t last train score: {train_scores[-1]:.3f}\n"
                f"best Eval score: {best_eval_score:.3f}\t last eval score: {eval_scores[-1]:.3f}\n"
                f"Perturbed Test score: {perturbed_test_score:.3f}\n"
                f"Clean Test score: {clean_test_score:.3f}\n"
                f"Perturbation efficiency: {(perturbed_test_score/clean_test_score):.3f}\n")
    print(x)
    logger.info(x)

