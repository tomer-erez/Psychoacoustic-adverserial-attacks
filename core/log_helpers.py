import sys

def avg_scores(scores):
    return sum(scores) / len(scores)

def log_epoch_metrics(epoch, num_epochs, train_ctc, eval_ctc_clean, eval_ctc_perturbed,
                      train_wer, eval_wer_clean, eval_wer_perturbed):
    print(f"{'=' * 60}")
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"{'Metric':<10} | {'Train':>10} | {'Eval Clean':>12} | {'Eval Perturbed':>16}")
    print(f"{'-' * 60}")
    print(f"{'CTC':<10} | {train_ctc:>10.0f} | {eval_ctc_clean:>12.0f} | {eval_ctc_perturbed:>16.0f}")
    print(f"{'WER':<10} | {train_wer:>10.2f} | {eval_wer_clean:>12.2f} | {eval_wer_perturbed:>16.2f}")
    print(f"{'=' * 60}\n")
    sys.stdout.flush()
    sys.stderr.flush()

def log_summary_metrics(args,
                        best_train_ctc,
                        best_train_wer,
                        train_scores,
                        clean_ctc_test,
                        clean_wer_test,
                        pert_ctc_test,
                        pert_wer_test):
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Perturbation norm type:':<30} {args.norm_type}")
    print(f"{'Perturbation size:':<30} {args.attack_size_string}")
    print("-" * 60)
    print(f"{'Metric':<20} | {'Clean Test':>15} | {'Perturbed Test':>15}")
    print(f"{'-' * 60}")
    print(f"{'CTC':<20} | {clean_ctc_test:>15.2f} | {pert_ctc_test:>15.2f}")
    print(f"{'WER':<20} | {clean_wer_test:>15.3f} | {pert_wer_test:>15.3f}")
    print("=" * 60)
    sys.stdout.flush()
    sys.stderr.flush()

def log_train_progress(batch_idx, total_batches, ctc_scores, wer_scores, times):
    print(f"batch: {batch_idx}/{total_batches},\t"
          f"avg CTC: {avg_scores(ctc_scores):.0f},\t"
          f"avg WER: {avg_scores(wer_scores):.3f},\t"
          f"avg time: {avg_scores(times):.2f}")
    sys.stdout.flush()
    sys.stderr.flush()