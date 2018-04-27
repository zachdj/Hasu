"""
The training script runs several agents in parallel, each of which update a shared neural network
"""

import numpy as np

from hasu.networks.AtariNet import AtariNet
from hasu.utils.preprocess import Preprocessor
from hasu.agents.a2c import A2CAgent

from pysc2.lib import features
from pysc2.lib import actions
from pysc2.env import sc2_env

import torch.optim as optim

_MAP_NAME = "DefeatRoaches"

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


def main(num_envs=1, step_mul=8, max_steps=160, rollout_steps=16, network_class=AtariNet, use_gpu=True,
         screen_resolution=84, minimap_resolution=64,
         gamma=0.99,                # discount factor for future rewards
         value_loss_weight=0.5,     # how much weight should the value loss carry?
         entropy_weight=1e-3,       # how much weight to assign to the entropy loss (higher weight => more exploration)
         grad_norm_limit=40,        # limit on how much we will adjust a weight per update
         learning_rate=7e-4,        # learning rate for AC network
         screen_features=DEFAULT_SCREEN_FEATURES, minimap_features=DEFAULT_MINIMAP_FEATURES,
         flat_features=DEFAULT_FLAT_FEATURES, action_space=DEFAULT_ACTION_SPACE):

    # create preprocessor for agents to use
    preprocessor = Preprocessor(screen_features, minimap_features, flat_features, use_gpu=use_gpu)

    # 3-tuples describing network inputs
    screen_size = (len(screen_features), screen_resolution, screen_resolution)
    mm_size = (len(minimap_features), minimap_resolution, minimap_resolution)
    flat_size = preprocessor.get_flat_size()

    # network used for generating policy and value estimations
    network = network_class(screen_size=screen_size, minimap_size=mm_size, flat_size=flat_size, num_actions=524)

    # optimization strategy for training the network
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    # create agents and environments in which the agents will act
    agents = []
    environments = []
    observations = []  # keeps track of the last observations taken from an environment
    rollouts = []
    for i in range(0, num_envs):
        agent = A2CAgent(network, preprocessor, action_space=action_space, gamma=gamma,
                         value_loss_weight=value_loss_weight, entropy_weight=entropy_weight, use_gpu=use_gpu)
        agents.append(agent)
        env = sc2_env.SC2Env(
              map_name=_MAP_NAME,
              agent_race=None,
              bot_race=None,
              difficulty=None,
              step_mul=step_mul,
              game_steps_per_episode=0,
              screen_size_px=(screen_resolution, screen_resolution),
              minimap_size_px=(minimap_resolution, minimap_resolution),
              visualize=False)
        environments.append(env)
        observations.append(env.reset()[0])
        rollout = {
            'action_masks': [],
            'policy': [],
            'value': [],
            'reward': []
        }
        rollouts.append(rollout)

    # iterate until max_steps reached
    step_counter = 0
    while step_counter < max_steps:
        step_counter += rollout_steps
        # rollout each agent
        for i in range(0, num_envs):
            agent, env, rollout = agents[i], environments[i], rollouts[i]
            # step forward n steps
            for step in range(0, rollout_steps):
                print(step)
                last_obs = observations[i]

                # print(last_obs)
                action = agent.step(last_obs)
                new_obs = env.step([action])
                observations[i] = new_obs[0]
            # TODO


if __name__ == '__main__':

    # hack to get pysc2 to accept flags correctly
    import sys
    from absl import flags

    FLAGS = flags.FLAGS
    FLAGS(sys.argv)

    # entry point
    main()
