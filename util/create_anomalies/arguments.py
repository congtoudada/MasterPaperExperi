import argparse


def get_arg_parser():
    parser = argparse.ArgumentParser('MAE abnormal', add_help=False)
    parser.add_argument("--ubnormal_path", type=str, default="H:/AI/dataset/VAD/Featurize/UBnormal")
    parser.add_argument("--input_dataset", type=str, default="H:/AI/dataset/VAD/Featurize/ShanghaiTech")
    parser.add_argument("--output_dataset", type=str, default="H:/AI/dataset/VAD/Featurize/ShanghaiTech/train/aug")
    parser.add_argument("--run_type", type=str, default="abnormal_objects")
    parser.add_argument('--target_size', default=(640, 320), type=int, help='images input size')
    args = parser.parse_args()
    return args
