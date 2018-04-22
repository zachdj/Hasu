"""
Entry point
"""
import numpy as np

from pysc2.lib import actions
from pysc2.lib import features

from hasu.networks.AtariNet import AtariNet

network = AtariNet()

for name, child in network.named_children():
    print(name)

# for arg in actions.TYPES:
#     print(arg.name)
#     for dim, size in enumerate(arg.sizes):
#         print(dim)
#         print(size)

# scale = 3
# numpy_feature = np.array([
#     [5, 1, 2],
#     [1, 2, 0],
#     [1, 1, 1],
# ])
#
# numpy_feature = numpy_feature[None, :, :]
#
# print(numpy_feature[:, 0, 0])
