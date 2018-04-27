"""
Entry point
"""
import numpy as np

from pysc2.lib import actions
from pysc2.lib import features

from hasu.networks.AtariNet import AtariNet

network = AtariNet()

ACTION_SPACE_INDICES = np.concatenate([
    np.arange(0, 39),  # attack, move, behavior actions
    [261],  # halt  (but don't catch fire)
    [274],  # hold position
    np.arange(331, 335),  # move screen, move minimap, and patrolling
], axis=0)
print(ACTION_SPACE_INDICES)
