import argparse

def create_arg_parser():
    parser = argparse.ArgumentParser()

    """standard training params"""
    parser.add_argument('--batch_size', type=int,default=128,help='batch size')
    parser.add_argument('--lr', type=float,default=1e-5,help='lr for the perturbation update')
    parser.add_argument('--early_stopping', type=int,default=5,help='how many epochs to wait before early stopping')

    """adverserial training specific params"""
    parser.add_argument('--norm_type', type=str,
                        choices=["l2", "snr", "fletcher_munson","linf","leakage"],
                        default='l2',help='type of norm to limit the perturbation')
    parser.add_argument('--pert_size', type=float,default=5,help='the size of the perturbation')

    """paths"""
    parser.add_argument('--save_path', type=str,
                        default='./save_data_dir',help='path_dir_to_save_data')
    parser.add_argument('--dataset_path', type=str,
                        default='./data',help="path to the dataset")
    parser.add_argument('--models_path', type=str,
                        default='./models',help='path to the speech recognition model')
    parser.add_argument('--model_to_attack', type=str,choices=['whisper','wav2vec','conformer_ctc'],
                        default='whisper',help='which model to attack')

    """others"""
    parser.add_argument('--seed', type=int,default=5,help='random seed for reproducibility')
    parser.add_argument('--jobid', type=str,
                        default='9999',help='job id')

    return parser