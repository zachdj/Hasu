"""
The run script runs a single agent using a saved neural network for a specified number of episodes
"""

import numpy as np

from hasu.networks.AtariNet import AtariNet
from hasu.utils.preprocess import Preprocessor
from hasu.agents.a2c import A2CAgent

from pysc2.lib import features
from pysc2.env import sc2_env

import torch

_MAP_NAME = "DefeatRoaches"

LIMITED_SCREEN_FEATURES = [
    features.SCREEN_FEATURES.player_relative,
    features.SCREEN_FEATURES.unit_type,
    features.SCREEN_FEATURES.selected,
    features.SCREEN_FEATURES.unit_hit_points,
    features.SCREEN_FEATURES.unit_hit_points_ratio,
    features.SCREEN_FEATURES.unit_density,
]

LIMITED_MINIMAP_FEATURES = [
    features.MINIMAP_FEATURES.player_relative,
    features.MINIMAP_FEATURES.selected
]

LIMITED_FLAT_FEATURES = [
    "player",           # A (11) tensor showing general information.
    "single_select",    # A (7) tensor showing information about a selected unit.
    "multi_select",     # (n, 7) tensor with the same as single select but for all n selected units
    "control_groups"    # (10, 2) tensor showing the (unit leader type and count) for each of the 10 control groups
]

# actions that we allow our agent to consider
LIMITED_ACTION_SPACE = np.zeros(524)
allowed_actions = np.concatenate([
    np.arange(0, 39),  # attack, move, behavior actions
    [261],  # halt  (but don't catch fire)
    [274],  # hold position
    np.arange(331, 335),  # move screen, move minimap, and patrolling
], axis=0)
LIMITED_ACTION_SPACE[allowed_actions] = 1


def main(step_mul=8, num_episodes=1, network_class=AtariNet, saved_model='./output/lim_obs_and_action/a2c_step5024.state',
         screen_resolution=84, minimap_resolution=64, use_gpu=True,
         limit_observation_space=False,  # use the limited set of screen, minimap, and flat features
         limit_action_space=False,       # use a limited set of available actions
     ):

    # hack to get pysc2 to accept flags correctly
    from absl import flags
    FLAGS = flags.FLAGS
    FLAGS(['Hasu'])

    # setup features to use
    screen_features = features.SCREEN_FEATURES
    minimap_features = features.MINIMAP_FEATURES
    flat_features = LIMITED_FLAT_FEATURES
    action_space = np.ones(524)
    if limit_observation_space:
        screen_features = LIMITED_SCREEN_FEATURES
        minimap_features = LIMITED_MINIMAP_FEATURES
    if limit_action_space:
        action_space = LIMITED_ACTION_SPACE

    # create preprocessor for agent to use
    preprocessor = Preprocessor(screen_features, minimap_features, flat_features, use_gpu=use_gpu)

    # 3-tuples describing network inputs
    screen_size = (len(screen_features), screen_resolution, screen_resolution)
    mm_size = (len(minimap_features), minimap_resolution, minimap_resolution)
    flat_size = preprocessor.get_flat_size()

    # network used for generating policy and value estimations
    network = network_class(screen_size=screen_size, minimap_size=mm_size, flat_size=flat_size, num_actions=524)

    if saved_model is not None:
        state = torch.load(saved_model)
        network.load_state_dict(state['state_dict'])

    # create agents and sc2 environment
    agent = A2CAgent(network, preprocessor, action_space=action_space, use_gpu=use_gpu)
    env = sc2_env.SC2Env(
        map_name=_MAP_NAME,
        agent_race=None,
        bot_race=None,
        difficulty=None,
        step_mul=step_mul,
        game_steps_per_episode=0,
        screen_size_px=(screen_resolution, screen_resolution),
        minimap_size_px=(minimap_resolution, minimap_resolution),
        visualize=True)

    # get the initial observation
    observation = env.reset()[0]

    # keep track of how many episodes the agent has completed
    episode_counter = 0

    # play the episodes
    while episode_counter < num_episodes:
        # step forward
        action = agent.step(observation)
        observation = env.step([action])[0]

        if observation.step_type == sc2_env.environment.StepType.LAST:
            episode_counter += 1
            observation = env.reset()[0]
            agent.reset()

    # report results
    average_reward = agent.reward / agent.episodes
    print(f'Agent loaded from {saved_model} ran for {num_episodes} episodes\n'
          f'Achieved an average reward of {average_reward}')

    env.close()


if __name__ == '__main__':
    main()