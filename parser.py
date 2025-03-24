import argparse

def create_arg_parser():
    parser = argparse.ArgumentParser()

    """standard training params"""
    parser.add_argument('--batch_size', type=int,default=2,help='batch size')
    parser.add_argument('--lr', type=float,default=5e-5,help='lr for the perturbation update')
    parser.add_argument('--early_stopping', type=int,default=3,help='how many epochs to wait before early stopping')
    parser.add_argument('--num_epochs', type=int,default=5,help='how many epochs at all')
    parser.add_argument('--optimize_type', type=str,
                        choices=["adam","pgd"],
                        default='pgd',help='how to optimize the perturbation update')
    parser.add_argument('--download_ds', action='store_true',
                        help='If set, download the whole dataset and cache it')


    """adverserial training specific params"""
    parser.add_argument('--norm_type', type=str,
                        choices=["l2","linf","snr",
                                  "fletcher_munson","leakage","min_max_freqs"],
                        default='fletcher_munson',help='type of norm to limit the perturbation')
    #sizes of perturbations
    parser.add_argument('--fm_epsilon', type=float,default=0.55,help='size of the fm_epsilon perturbation')
    parser.add_argument('--l2_size', type=float,default=0.1,help='size of the l2 perturbation')
    parser.add_argument('--linf_size', type=float,default=0.0008,help='size of the linf perturbation')
    parser.add_argument('--snr_db', type=int,default=48,help='size of the signal to noise ratio ')
    parser.add_argument('--min_freq_attack', type=int,default=800,help='min freq to perturb in, remember that the human threshold is ~20 hz')
    parser.add_argument('--max_freq_attack', type=int,default=2e4,help='max freq to perturb in, remember that the human threshold is ~20000 hz')
    parser.add_argument('--min_freq_leakage', type=int,default=20,help='min freq to attack in leakage meaning the humans exact frequency boundaries, remember that the human threshold is ~20 hz')
    parser.add_argument('--max_freq_leakage', type=int,default=2e4,help='max freq to attack in leakage meaning the humans exact frequency boundaries, remember that the human threshold is ~20000 hz')


    """paths"""
    parser.add_argument('--dataset_path', type=str,
                        default='./train-clean-100',help="path to the dataset")

    """sound properties"""
    parser.add_argument('--sr', type=int,default=16000,help='sample rate')
    parser.add_argument('--n_fft', type=int,default=2048,help='numbers of FFT bins for stft')

    """others"""
    parser.add_argument('--seed', type=int,default=5,help='random seed for reproducibility')
    parser.add_argument('--jobid', type=str,
                        default='9999',help='job id')
    parser.add_argument('--report_interval', type=int,default=100,help='report interval')
    parser.add_argument('--small_data', action='store_true',
                        help='If set, use only 1% of the dataset for fast debugging')
    parser.add_argument('--num_items_to_inspect', type=int,default=6,help='number of items to inspect, visualize')


    return parser