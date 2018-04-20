"""
AtariNet network from https://arxiv.org/abs/1708.04782
"""

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from pysc2.lib import actions


# the output size of the linear layer connected to the non-spatial inputs
# This parameter is not specified in the DeepMind paper; 32 seems to be a common choice
_STRUCTURED_OUTPUT_SIZE = 32

# units in the fully connected layer after concatenating features
_FC_OUTPUT_SIZE = 256


class AtariNet(nn.Module):
    def __init__(self, minimap_size=(2, 64, 64), screen_size=(5, 84, 84), flat_size=2775):
        """ Initialize the network

        Args:
            minimap_size (tuple): 3-tuple (channels, width, height) specifying the size of the minimap input
            screen_size (tuple): 3-tuple (channels, width, height) specifying the size of the screen input
            flat_size (int): size of the structured features vector
        """
        super().__init__()

        self.minimap_features = nn.Sequential(
            nn.Conv2d(in_channels=minimap_size[0], out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
        )

        self.screen_features = nn.Sequential(
            nn.Conv2d(in_channels=screen_size[0], out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(inplace=True)
        )

        self.flat_features = nn.Sequential(
            nn.Linear(flat_size, _STRUCTURED_OUTPUT_SIZE),
            nn.Tanh()
        )

        # self.minimap_conv1 = nn.Conv2d(in_channels=minimap_size[0], out_channels=16, kernel_size=8, stride=4)
        # self.minimap_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        #
        # self.screen_conv1 = nn.Conv2d(in_channels=screen_size[0], out_channels=16, kernel_size=8, stride=4)
        # self.screen_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)

        # self.flat_fc = nn.Linear(flat_size, _STRUCTURED_OUTPUT_SIZE)

        # The output of the above layes are concatenated and sent through a fully connected layer
        # we need to find the size of the concatenated tensor (it will vary based on mm and screen size)
        # the easiest and most accurate way is to send a random tensor through the network and check the output size
        example = Variable(torch.rand(1, *minimap_size))
        example = self.minimap_features(example)
        minimap_output_size = example.data.view(1, -1).size(1)

        example = Variable(torch.rand(1, *screen_size))
        example = self.screen_features(example)
        screen_output_size = example.data.view(1, -1).size(1)

        total_size = minimap_output_size + screen_output_size + _STRUCTURED_OUTPUT_SIZE

        # The results are concatenated and sent through a linear layer with a ReLU activation.
        self.combined_features = nn.Sequential(
            nn.Linear(total_size, _FC_OUTPUT_SIZE),
            nn.ReLU()
        )

        # Linear layer that predicts the value function from the latent vector
        self.value_predictor = nn.Linear(_FC_OUTPUT_SIZE, 1)

        # Linear Layer that computes a distribution over the action space
        self.policy_action = nn.Sequential(
            nn.Linear(_FC_OUTPUT_SIZE, 524),  # there are 524 available action functions
            nn.Softmax()
        )

        """
        Linear Layers that compute independent distributions over each function argument
        Explanation: each function call takes some number of arguments.  Each of the arguments is one of the 13 types
        defined in pysc2.lib.actions.TYPES
        Each arg has one or more dimensions, and each dimension has some "size" specifying the number of choices for 
        that arg.
        For example, the "screen" argument type has two dimensions (x and y), 
        each of which take a value in the range [0, screen_size)
        
        We create a linear layer for each dimension of each argument type, with hidden units == <size>
        """
        self.policy_args_fc = dict()
        for arg in actions.TYPES:
            self.policy_args_fc[arg.name] = dict()
            for dim, size in enumerate(arg.sizes):
                if arg.name == 'screen' or arg.name == 'screen2':
                    arg_size = screen_size[1]
                elif arg.name == 'minimap':
                    arg_size = minimap_size[1]
                else:
                    arg_size = size
                self.policy_args_fc[arg.name][dim] = nn.Sequential(
                    nn.Linear(_FC_OUTPUT_SIZE, arg_size),
                    nn.Softmax()
                )

    def forward(self, screen, minimap, flat, available_actions):
        """ Pushes an observation through the network and computes value estimation and a choice of action

        Args:
            screen (autograd.Variable): Variable with N x C x W x H screen features
            minimap (autograd.Variable): Variable with N x C x W x H minimap features
            flat (autograd.Variable): Variable with N x C flat (structured) features
            available_actions (ndarray): mask for available actions

        Returns:
            (policy, value) tuple

            `policy` will be a tuple (function_id, args) where `function_id` contains a distribution over the action space,
                and `args` is a dictionary containing a distribution over the choices for each dimension of each arg type

            `value` will be a scalar estimation of the value of the current state
        """
        # push each input through the network
        screen = self.screen_features(screen)
        minimap = self.minimap_features(screen)
        flat = self.flat_features(flat)

        flattened_screen = screen.view(1, -1)
        flattened_mm = minimap.view(1, -1)

        latent_vector = torch.cat([flat, flattened_screen, flattened_mm], 1)
        # latent_vector = self.combined_features(latent_vector)

        print(latent_vector.shape)
