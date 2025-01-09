import argparse

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comments', default="", type=str,
                        help='comments to write in the logger, this is your place to highlight important changes in the current run')
    parser.add_argument('--room_edges_path', type=str,
                        default='/home/tomer.erez/EARS/data/shifted_32_angle_inr_data_with_frl_apartment_2/edge_points_array.txt',
                        help='txt file of the room edges')
    parser.add_argument('--room_min_max_coords_path', type=str,
                        default='/home/tomer.erez/EARS/data/shifted_32_angle_inr_data_with_frl_apartment_2/frl_apartment_2_min_max_coords.txt',
                        help='txt file of the room edges')
    return parser