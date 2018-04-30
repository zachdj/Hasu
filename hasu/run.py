"""
The run script runs a single agent using a saved neural network for a specified number of episodes
"""

import torch

from hasu.networks.AtariNet import AtariNet


def main():
    # TODO
    # load network
    network = AtariNet()
    network.load_state_dict(torch.load('../output/a2c_step1584.network'))
    network.eval()


if __name__ == '__main__':
    main()