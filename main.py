import build
import eval
import parser
import train
import save
import torch

if __name__ == '__main__':
    args = parser.create_arg_parser().parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger=build.create_logger(args=args)
    train_data_loader, eval_data_loader,test_data_loader,audio_length = build.create_data_loaders(args=args,logger=logger)
    model,processor = build.load_model(args)
    p = build.init_perturbation(args,length=audio_length)
    optimizer = build.create_perturbation_optimizer(args=args, p=p)

    train_scores = []
    eval_scores = []
    no_improve_epochs=0
    best_eval_score = -1*float('inf')

    for epoch in range(args.num_epochs):
        p, train_epoch_score = train.train_epoch(
            args=args, train_data_loader=train_data_loader, p=p,
            model=model, optimizer=optimizer,
            epoch=epoch, logger=logger, processor=processor
        )
        train_scores.append(train_epoch_score)
        eval_epoch_score = eval.evaluate(
            args=args, eval_data_loader=eval_data_loader, p=p,
            model=model, logger=logger,
            perturbed=True, epoch_number=epoch, processor=processor
        )
        eval_scores.append(eval_epoch_score)

        if eval_epoch_score>best_eval_score:#update
            no_improve_epochs,best_eval_score=0,eval_epoch_score
        else:
            no_improve_epochs=no_improve_epochs+1

        if no_improve_epochs>args.patience:#early stopping
            logger.info(f'no improvements in {no_improve_epochs} epochs, stopping training and testing on the test set')
            break

    perturbed_test_score = eval.evaluate( #eval perturbation
        args=args, eval_data_loader=test_data_loader, p=p,
        model=model, logger=logger,
        perturbed=True, epoch_number=-1, processor=processor
    )
    clean_test_score = eval.evaluate( #eval clean model
        args=args, eval_data_loader=test_data_loader, p=torch.zeros_like(p),
        model=model, logger=logger,
        perturbed=False, epoch_number=-1, processor=processor
    )
    logger.info(f"Summary:\n"
                f"Perturbation norm type: {args.norm_type}\t"
                f"Perturbation size: {args.pert_size}\n"
                f"Train score: {max(train_scores):.4f}\n"
                f"Eval score: {best_eval_score:.4f}\n"
                f"Perturbed Test score: {perturbed_test_score:.4f}\n"
                f"Clean Test score: {clean_test_score:.4f}\n"
                f"Perturbation efficiency: {(perturbed_test_score/clean_test_score):.4f}\n")

    save.save_perturbation(args,p)
