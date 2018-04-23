"""
A2C Agent

Learns to play starcraft minigames using Synchronous Advantage Actor-Critic (A2C)

References:
    [1] Schulman, John, et al. "High-dimensional continuous control using generalized advantage estimation."
        arXiv preprint arXiv:1506.02438 (2015).
"""

import time

import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import torch
from torch.autograd import Variable

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

# actions that we allow our agent to consider
DEFAULT_ACTION_SPACE = np.zeros(524)
allowed_actions = np.concatenate([
    np.arange(0, 39),  # attack, move, behavior actions
    [261],  # halt  (but don't catch fire)
    [274],  # hold position
    np.arange(331, 335),  # move screen, move minimap, and patrolling
], axis=0)
DEFAULT_ACTION_SPACE[allowed_actions] = 1


class A2CAgent(base_agent.BaseAgent):

    def __init__(self,
                 screen_features=DEFAULT_SCREEN_FEATURES,
                 minimap_features=DEFAULT_MINIMAP_FEATURES,
                 flat_features=DEFAULT_FLAT_FEATURES,
                 action_space=DEFAULT_ACTION_SPACE,  # binary mask of length 524
                 screen_size=84,
                 minimap_size=64,
                 gamma=0.99,  # discount factor for future rewards
                 value_loss_weight=0.5,
                 entropy_weight=1e-3,
                 learning_rate=7e-4,
                 use_gpu=True):

        super().__init__()
        self.screen_features = screen_features
        self.minimap_features = minimap_features
        self.flat_features = flat_features
        self.action_space = action_space
        self.action_space_variable = Variable(torch.from_numpy(self.action_space).float())
        self.gamma = gamma
        self.value_loss_weight = value_loss_weight
        self.entropy_weight = entropy_weight
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu

        screen_size = (len(screen_features), screen_size, screen_size)
        mm_size = (len(minimap_features), minimap_size, minimap_size)

        self.preprocessor = Preprocessor(self.screen_features, self.minimap_features, self.flat_features, use_gpu=use_gpu)

        flat_size = self.preprocessor.get_flat_size()

        self.network = AtariNet(screen_size=screen_size, minimap_size=mm_size,
                                flat_size=flat_size, num_actions=524)
        if use_gpu:
            self.network = self.network.cuda()
            self.action_space_variable = self.action_space_variable.cuda()

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

        policy_action, policy_args, value = self.network(screen, minimap, flat, available_actions)
        network_time = time.time()

        # select an action according to the policy probabilities
        # mask unavailable actions and actions that aren't in our agent's action space
        action_distribution = policy_action * available_actions * self.action_space_variable
        action_distribution = torch.div(action_distribution, torch.sum(action_distribution))  # renormalize

        np_action_distribution = action_distribution.data.cpu().numpy()[0]
        selected_action_id = np.random.choice(np.arange(0, len(actions.FUNCTIONS)), p=np_action_distribution)
        print("\nChose action %s" % selected_action_id)
        print("with probability:  %s\n" % np_action_distribution[selected_action_id])

        # select arguments from the argument outputs
        args = []
        arg_distributions = dict()  # passed to the loss function
        for arg in actions.FUNCTIONS[selected_action_id].args:
            selected_values = np.zeros(len(arg.sizes), dtype=np.int)  # arg can have multiple dimensions
            for dim, size in enumerate(arg.sizes):
                arg_module_name = AtariNet.get_argument_module_name(arg, dim)
                # distribution over possible argument values:
                arg_distribution = policy_args[arg_module_name].data.cpu().numpy()[0]
                selected_val = np.random.choice(np.arange(0, len(arg_distribution)), p=arg_distribution)
                selected_values[dim] = selected_val

                # save the distribution for the loss function
                arg_distributions[arg_module_name] = policy_args[arg_module_name]
            # add the argument to the list of args passed to pysc2
            args.append(selected_values)
        print("\nChose args: %s\n" % args)

        # print("preprocess time: %0.6f s" % (preproc_time - start_time))
        # print("network time: %0.6f s" % (network_time - preproc_time))

        action_mask_rollout = [available_actions]
        policy_rollout = [(policy_action, policy_args, selected_action_id, args)]
        value_estimation_rollout = [value]
        reward_rollout = [np.float(obs.reward), 10.]

        loss = self.compute_loss(action_mask_rollout, policy_rollout, value_estimation_rollout, reward_rollout)
        print("\n\n Loss: %s \n\n" % loss)

        # return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
        return actions.FunctionCall(selected_action_id, args)

    def compute_loss(self, action_mask_rollout, policy_rollout, value_estimate_rollout, rewards):
        """ Computes loss function used for backprop from a rollout sequence

        Uses advantage estimate from [1].
        Influenced by https://github.com/ikostrikov/pytorch-a3c and
        https://github.com/Jiankai-Sun/Asynchronous-Advantage-Actor-Critic-in-PyTorch

        Args:
            action_mask_rollout: sequence of n `available_action` masks
            policy_rollout: sequence of n tuples (action_probs, argument_probabilities, selected_action_id, selected_args)
            value_estimate_rollout: sequence n of estimated values
            rewards: sequence of n+1 rewards received from the env

        Returns:
            torch.autograd.Variable containing the loss from the sequence

        """
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        if self.use_gpu:
            gae = gae.cuda()
        R = value_estimate_rollout[-1]  # bootstrap with the most recent value estimate
        for t in reversed(range(len(policy_rollout))):  # loop backwards through time
            R = self.gamma * R + rewards[t]
            advantage = R - value_estimate_rollout[t]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation from [1]
            if t != (len(policy_rollout) - 1):
                delta_t = rewards[t] + self.gamma * value_estimate_rollout[t + 1].data - value_estimate_rollout[t].data
            else:
                delta_t = rewards[t] + self.gamma * rewards[t + 1] - value_estimate_rollout[t].data
            gae = gae * self.gamma + delta_t

            # compute log prob(a | s)
            policy = policy_rollout[t]
            action_distribution, selected_action = policy[0], policy[2]
            masked_distribution = action_distribution * action_mask_rollout[t] * self.action_space_variable
            masked_distribution = torch.div(masked_distribution, torch.sum(masked_distribution))
            action_log_prob = torch.log(masked_distribution[0][selected_action])

            argument_distributions, selected_args = policy[1], policy[3]
            for idx, arg in enumerate(actions.FUNCTIONS[selected_action].args):
                for dim, size in enumerate(arg.sizes):
                    arg_module_name = AtariNet.get_argument_module_name(arg, dim)
                    arg_dist = argument_distributions[arg_module_name]
                    arg_value = selected_args[idx][dim]
                    action_log_prob += torch.log(arg_dist[0][arg_value])

            # compute entropy loss
            clamped_probs = torch.clamp(masked_distribution[0], 1e-5, 0.99999)  # logs don't deal with 0s and 1s very well
            log_probs = torch.log(clamped_probs)
            entropy = torch.dot(log_probs, masked_distribution[0])

            policy_loss = policy_loss - action_log_prob * Variable(gae) - self.entropy_weight * entropy

        return self.value_loss_weight * value_loss + policy_loss
