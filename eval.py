import torch

def evaluate(args, eval_data_loader, p, model, criterion, logger, perturbed, epoch_number):
    model.eval()
    scores = []

    with torch.no_grad():  # <-- THIS IS WHAT YOU WANT
        for data, labels in eval_data_loader:
            data = data + p if perturbed else data
            y_pred = model(data)
            loss = criterion(y_pred, labels)
            scores.append(loss.item())

    avg_score = sum(scores) / len(scores) if scores else float('inf')

    if epoch_number == -1:
        msg = f"TEST set: {'PERTURBED' if perturbed else 'CLEAN'}:\t avg score {avg_score:.4f}\n\n"
    else:
        msg = f"Evaluation set: Epoch {epoch_number}, avg score {avg_score:.4f}\n\n"

    logger.info(msg)
    return avg_score
