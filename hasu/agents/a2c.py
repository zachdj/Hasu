"""
A2C Agent

Learns to play starcraft minigames using Synchronous Advantage Actor-Critic (A2C)
"""

import time

import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from hasu.utils import constants
from hasu.networks.AtariNet import AtariNet
from hasu.utils.preprocess import Preprocessor

# define the default features that this agent uses
# these are documented at https://github.com/deepmind/pysc2/blob/master/docs/environment.md
DEFAULT_SCREEN_FEATURES = [
    features.SCREEN_FEATURES.player_relative,
    features.SCREEN_FEATURES.unit_type,
    features.SCREEN_FEATURES.selected,
    features.SCREEN_FEATURES.unit_hit_points,
    features.SCREEN_FEATURES.unit_hit_points_ratio,
    features.SCREEN_FEATURES.unit_density,
]

DEFAULT_MINIMAP_FEATURES = [
    features.MINIMAP_FEATURES.player_relative,
    features.MINIMAP_FEATURES.selected
]

DEFAULT_FLAT_FEATURES = [
    "player",           # A (11) tensor showing general information.
    "single_select",    # A (7) tensor showing information about a selected unit.
    "multi_select",     # (n, 7) tensor with the same as single select but for all n selected units
    "control_groups"    # (10, 2) tensor showing the (unit leader type and count) for each of the 10 control groups
]


class A2CAgent(base_agent.BaseAgent):

    def __init__(self, screen_features=DEFAULT_SCREEN_FEATURES, minimap_features=DEFAULT_MINIMAP_FEATURES,
                 flat_features=DEFAULT_FLAT_FEATURES, screen_size=84, minimap_size=64, use_gpu=True):
        super().__init__()
        self.screen_features = screen_features
        self.minimap_features = minimap_features
        self.flat_features = flat_features
        self.use_gpu = use_gpu

        screen_size = (len(screen_features), screen_size, screen_size)
        mm_size = (len(minimap_features), minimap_size, minimap_size)

        self.preprocessor = Preprocessor(self.screen_features, self.minimap_features, self.flat_features, use_gpu=use_gpu)

        flat_size = self.preprocessor.get_flat_size()

        self.network = AtariNet(screen_size=screen_size, minimap_size=mm_size, flat_size=flat_size)
        if use_gpu:
            self.network = self.network.cuda()

    def step(self, obs):
        """ Takes an observation from the environment and returns an action to perform

        Args:
            obs: an observation from the pysc2 environment

        Returns:
            pysc2.lib.actions.FunctionCall object specifying the action to take

        """
        super(A2CAgent, self).step(obs)
        start_time = time.time()

        screen, minimap, flat, available_actions = self.preprocessor.process(obs.observation)
        preproc_time = time.time()

        action, policy_args, value = self.network(screen, minimap, flat, available_actions)
        network_time = time.time()

        action_mask = available_actions.data.cpu().numpy()  # array with ones for available actions and zeros otherwise
        action_distribution = action.data.cpu().numpy()[0]
        action_distribution = action_distribution * action_mask  # set probability of invalid actions to zero
        action_distribution = action_distribution / np.sum(action_distribution)  # renormalize

        # select an action id from the distribution returned from the network
        action_id = np.random.choice(np.arange(len(action_distribution)), p=action_distribution)
        print("\nChose action %s\n" % action_id)

        # select arguments from the argument outputs
        args = []
        for arg in actions.FUNCTIONS[action_id].args:
            selected_values = np.zeros(len(arg.sizes), dtype=np.int)  # value can have multiple dimensions
            for dim, size in enumerate(arg.sizes):
                arg_module = AtariNet.get_argument_module_name(arg, dim)
                arg_dist = policy_args[arg_module].data.cpu().numpy()
                selected_val = np.argmax(arg_dist)
                selected_values[dim] = selected_val
            args.append(selected_values)
        print("\nChose args: %s\n" % args)

        # print("preprocess time: %0.6f s" % (preproc_time - start_time))
        # print("network time: %0.6f s" % (network_time - preproc_time))

        # return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
        return actions.FunctionCall(action_id, args)
