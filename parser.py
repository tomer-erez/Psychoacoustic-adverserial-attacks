import argparse

def create_arg_parser():
    parser = argparse.ArgumentParser()

    """standard training params"""
    parser.add_argument('--batch_size', type=int,default=10,help='batch size')
    parser.add_argument('--lr', type=float,default=4e-4,help='lr for the perturbation update')
    parser.add_argument('--early_stopping', type=int,default=3,help='how many epochs to wait before early stopping')
    parser.add_argument('--num_epochs', type=int,default=2,help='how many epochs at all')
    parser.add_argument('--optimize_type', type=str,
                        choices=["adam","pgd"],
                        default='pgd',help='how to optimize the perturbation update')
    parser.add_argument('--download_ds', action='store_true',
                        help='If set, download the whole dataset and cache it')


    """adverserial training specific params"""
    parser.add_argument('--target_reps', type=int,default=10,
                        help='how many times should the model predict the target word, for example if 1: <delete>, if 5: <delete delete delete delete delete> as the label to optimize')
    parser.add_argument('--target', type=str,
                        default="delete",help='Target phrase for targeted attacks')
    parser.add_argument('--attack_mode', type=str,
                        choices=["untargeted", "targeted"],
                        default="untargeted",help='if set, we train a perturbation to increase the likeleyhood of predicting this word')
    parser.add_argument('--loss_fn', type=str,
                        choices=["ctc", "wer"],
                        default="ctc",help='loss function to optimize the perturbation')



    parser.add_argument('--norm_type', type=str,
                        choices=["l2","linf","snr",
                                  "fletcher_munson","leakage","min_max_freqs"],
                        default='snr',help='type of norm to limit the perturbation')
    #sizes of perturbations
    parser.add_argument('--fm_epsilon', type=float,default=2,help='size of the fm_epsilon perturbation')
    parser.add_argument('--l2_size', type=float,default=0.09,help='size of the l2 perturbation')
    parser.add_argument('--linf_size', type=float,default=0.0001,help='size of the linf perturbation')
    parser.add_argument('--snr_db', type=int,default=40,help='size of the signal to noise ratio ')
    parser.add_argument('--min_freq_attack', type=int,default=400,help='min freq to perturb in, remember that the human threshold is ~20 hz')
    parser.add_argument('--max_freq_attack', type=int,default=2e4,help='max freq to perturb in, remember that the human threshold is ~20000 hz')
    parser.add_argument('--min_freq_leakage', type=int,default=20,help='min freq to attack in leakage meaning the humans exact frequency boundaries, remember that the human threshold is ~20 hz')
    parser.add_argument('--max_freq_leakage', type=int,default=2e4,help='max freq to attack in leakage meaning the humans exact frequency boundaries, remember that the human threshold is ~20000 hz')


    """paths"""
    parser.add_argument('--dataset_path', type=str,
                        default='./train-clean-100',help="path to the dataset")
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to a saved perturbation .pt file to resume training from')

    """sound properties"""
    parser.add_argument('--sr', type=int,default=16000,help='sample rate')
    parser.add_argument('--n_fft', type=int,default=2048,help='numbers of FFT bins for stft')

    """others"""
    parser.add_argument('--seed', type=int,default=5,help='random seed for reproducibility')
    parser.add_argument('--jobid', type=str,
                        default='9999',help='job id')
    parser.add_argument('--report_interval', type=int,default=200,help='report interval')
    parser.add_argument('--small_data', action='store_true',
                        help='If set, use only 1% of the dataset for fast debugging')
    parser.add_argument('--num_items_to_inspect', type=int,default=10,help='number of items to inspect, visualize')
    parser.add_argument('--run_all_norms', action='store_true',
                        help='If set, run 1 script for each norm, just a for loop on the main function')


    return parser