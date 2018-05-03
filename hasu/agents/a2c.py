"""
A2C Agent

Learns to play starcraft minigames using Synchronous Advantage Actor-Critic (A2C)

References:
    [1] Schulman, John, et al. "High-dimensional continuous control using generalized advantage estimation."
        arXiv preprint arXiv:1506.02438 (2015).
"""
import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import torch
from torch.autograd import Variable

from hasu.networks.AtariNet import AtariNet
from hasu.utils.preprocess import Preprocessor


_DEFAULT_NETWORK = AtariNet()
_DEFAULT_PREPROCESSOR = Preprocessor(features.SCREEN_FEATURES, features.MINIMAP_FEATURES,
                                     ["player", "single_select", "multi_select", "control_groups"], use_gpu=True)

_STOCHASTIC_CHOICE_PCT = 0.25


class A2CAgent(base_agent.BaseAgent):
    def __init__(self,
                 network=_DEFAULT_NETWORK,
                 preprocessor=_DEFAULT_PREPROCESSOR,
                 action_space=np.ones(524),  # binary mask of length 524
                 train=False,  # are we training the agent?
                 gamma=0.99,  # discount factor for future rewards
                 value_loss_weight=0.5,
                 entropy_weight=1e-3,
                 use_gpu=True):

        super().__init__()
        self.preprocessor = preprocessor
        self.network = network
        self.train = train

        # actions that our agent is allowed to select
        self.action_space = action_space
        self.action_space_variable = Variable(torch.from_numpy(self.action_space).float())

        # hyperparams
        self.gamma = gamma
        self.value_loss_weight = value_loss_weight
        self.entropy_weight = entropy_weight

        # keep track of rollouts
        self.rollout = {
            'action_mask': [],
            'policy': [],
            'value': [],
            'reward': []
        }

        # perform computations on the GPU?
        self.use_gpu = use_gpu
        if use_gpu:
            self.network = self.network.cuda()
            self.action_space_variable = self.action_space_variable.cuda()

    def reset(self):
        super(A2CAgent, self).reset()
        self.clear_rollout()

    def step(self, obs):
        """ Takes an observation from the environment and returns an action to perform

        Args:
            obs: an observation from the pysc2 environment

        Returns:
            pysc2.lib.actions.FunctionCall object specifying the action to take

        """
        super(A2CAgent, self).step(obs)

        screen, minimap, flat, available_actions = self.preprocessor.process(obs.observation)

        policy_action, policy_args, value = self.network(screen, minimap, flat, available_actions)

        # select an action according to the policy probabilities
        # mask unavailable actions and actions that aren't in our agent's action space
        action_distribution = policy_action * available_actions * self.action_space_variable
        action_distribution = torch.div(action_distribution, torch.sum(action_distribution))  # renormalize

        np_action_distribution = action_distribution.data.cpu().numpy()[0]
        # if training, choose actions stochastically
        if self.train:
            selected_action_id = np.random.choice(np.arange(0, len(actions.FUNCTIONS)), p=np_action_distribution)
        else:
            # if testing, only choose actions stochastically some of the time
            stochastic = np.random.random() < _STOCHASTIC_CHOICE_PCT
            if stochastic:
                selected_action_id = np.random.choice(np.arange(0, len(actions.FUNCTIONS)), p=np_action_distribution)
            else:
                selected_action_id = np.argmax(np_action_distribution)

        # select arguments from the argument outputs
        args = []
        arg_distributions = dict()  # passed to the loss function
        for arg in actions.FUNCTIONS[selected_action_id].args:
            selected_values = np.zeros(len(arg.sizes), dtype=np.int)  # arg can have multiple dimensions
            for dim, size in enumerate(arg.sizes):
                arg_module_name = AtariNet.get_argument_module_name(arg, dim)
                # distribution over possible argument values:
                arg_distribution = policy_args[arg_module_name].data.cpu().numpy()[0]
                if self.train:
                    selected_val = np.random.choice(np.arange(0, len(arg_distribution)), p=arg_distribution)
                else:
                    stochastic = np.random.random() < _STOCHASTIC_CHOICE_PCT
                    if stochastic:
                        selected_val = np.random.choice(np.arange(0, len(arg_distribution)), p=arg_distribution)
                    else:
                        selected_val = np.argmax(arg_distribution)

                selected_values[dim] = selected_val

                # save the distribution for the loss function
                arg_distributions[arg_module_name] = policy_args[arg_module_name]
            # add the argument to the list of args passed to pysc2
            args.append(selected_values)

        if self.train:
            # keep track of rollout
            self.rollout['action_mask'].append(available_actions)
            self.rollout['policy'].append((policy_action, policy_args, selected_action_id, args))
            self.rollout['value'].append(value)
            self.rollout['reward'].append(np.float(obs.reward))

        return actions.FunctionCall(selected_action_id, args)

    def compute_loss(self):
        """ Computes loss function used for backprop from a rollout sequence

        Uses advantage estimate from [1].
        Influenced by https://github.com/ikostrikov/pytorch-a3c and
        https://github.com/Jiankai-Sun/Asynchronous-Advantage-Actor-Critic-in-PyTorch

        Uses the rollout accumulated by the last several calls to the step function:
            action_mask_rollout: sequence of n `available_action` masks
            policy_rollout: sequence of n tuples (action_probs, argument_probabilities, selected_action_id, selected_args)
            value_estimate_rollout: sequence n of estimated values
            rewards: sequence of n rewards received from the env

        Returns:
            torch.autograd.Variable containing the loss from the sequence,
            or None if loss cannot be computed from the current rollout

        """
        action_mask_rollout = self.rollout['action_mask'][:-1]
        policy_rollout = self.rollout['policy'][:-1]
        value_estimate_rollout = self.rollout['value'][:-1]
        rewards = self.rollout['reward']

        # handle edge case where rollouts were collected for only one step:
        if len(policy_rollout) == 0:
            return None

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

    def clear_rollout(self):
        self.rollout = {
            'action_mask': [],
            'policy': [],
            'value': [],
            'reward': []
        }
