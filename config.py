import argparse
import torch


def parse_option():

    parser = argparse.ArgumentParser()

    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--head_size', type=int, default=64)
    parser.add_argument('--n_embed', type=int, default=512)
    parser.add_argument('--n_features', type=int, default=88)
    parser.add_argument('--n_layer', type=int, default=8)
    parser.add_argument('--device', type=torch.device, default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)

    opt = parser.parse_args()

    return opt