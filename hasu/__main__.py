"""
Entry point
"""
import numpy as np

from pysc2.lib import actions
from pysc2.lib import features

scale = 3
numpy_feature = np.array([
    [5, 1, 2],
    [1, 2, 0],
    [1, 1, 1],
])

numpy_feature = numpy_feature[None, :, :]

print(numpy_feature[:, 0, 0])
