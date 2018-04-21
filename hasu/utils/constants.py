"""
Module that exposes useful global constants
"""

from pysc2.lib import actions

# 200 seems to be the maximum number of units controllable by a single player (as imposed by supply cap)
# this could probably change in custom games
MAX_CONTROLLABLE_UNITS = 200

MAX_CARGO_SIZE = 8  # warp prisms and medivacs can carry up to 8 units

# gives the maximum flattened sizes of the structured features exposed by pysc2
# keys match the name of the structured feature in the obs.observation dictionary
MAX_FLAT_SIZES = {
    'player': 11,
    'control_groups': 20,
    'single_select': 7,
    'multi_select': MAX_CONTROLLABLE_UNITS * 7,
    'build_queue': 30 * 7,  # set somewhat arbitrarily
    'cargo': MAX_CARGO_SIZE * 7
}
