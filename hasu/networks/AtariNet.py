"""
AtariNet network from https://arxiv.org/abs/1708.04782
"""

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
    def __init__(self, screen_size=(5, 84, 84), minimap_size=(2, 64, 64), flat_size=1438, num_actions=524):
        """ Initialize the network

        Args:
            minimap_size (tuple): 3-tuple (channels, width, height) specifying the size of the minimap input
            screen_size (tuple): 3-tuple (channels, width, height) specifying the size of the screen input
            flat_size (int): size of the structured features vector
            use_gpu (bool): if set, computations will be performed on the current cuda device
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
            nn.Linear(_FC_OUTPUT_SIZE, num_actions),
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
        
        We add each linear layer as a child module.  The child module is accessible by its name using getattr(self, names)
        """
        for arg in actions.TYPES:
            for dim, size in enumerate(arg.sizes):
                if arg.name == 'screen' or arg.name == 'screen2':
                    arg_size = screen_size[1]
                elif arg.name == 'minimap':
                    arg_size = minimap_size[1]
                else:
                    arg_size = size

                module_name = self.get_argument_module_name(arg, dim)
                self.add_module(module_name, nn.Sequential(
                    nn.Linear(_FC_OUTPUT_SIZE, arg_size),
                    nn.Softmax()
                ))

    def forward(self, screen, minimap, flat, available_actions):
        """ Pushes an observation through the network and computes value estimation and a choice of action

        Args:
            screen (autograd.Variable): Variable with N x C x W x H screen features
            minimap (autograd.Variable): Variable with N x C x W x H minimap features
            flat (autograd.Variable): Variable with N x C flat (structured) features
            available_actions (ndarray): mask for available actions

        Returns:
            (action, args, value) tuple

            `action` contains a distribution over the action space,
            `args` is a dictionary with a distribution over valid argument values for each argument,
            `value` will be a scalar estimation of the value of the current state
        """
        # push each input through the network
        screen = self.screen_features(screen)
        minimap = self.minimap_features(minimap)
        flat = self.flat_features(flat)

        flattened_screen = screen.view(1, -1)
        flattened_mm = minimap.view(1, -1)

        latent_vector = torch.cat([flat, flattened_screen, flattened_mm], 1)
        features = self.combined_features(latent_vector)

        value = self.value_predictor(features)
        action = self.policy_action(features)

        policy_args = dict()
        for arg in actions.TYPES:
            for dim, size in enumerate(arg.sizes):
                module_name = self.get_argument_module_name(arg, dim)
                operator = getattr(self, module_name)
                policy_args[module_name] = operator(features)

        return action, policy_args, value

    @staticmethod
    def get_argument_module_name(arg, dim):
        """ Generates a unique name for the module that picks a value for the nth dimension of the given argument """
        return "arg_%s_dim%s" % (arg.name, dim)
