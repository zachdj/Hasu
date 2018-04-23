""" Functions for preprocessing pysc2 observations

TODO:
    * The screen/minimap processing isn't very DRY
    * We currently embed all continuous features using a single output channel.
        This may not be enough

"""

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from pysc2.lib import actions
from pysc2.lib import features

from hasu.utils import constants


class Preprocessor(object):
    """ Handles feature embedding and transforming
    """
    def __init__(self, screen_features, minimap_features, flat_features, use_gpu=True):
        """ Create a new Preprocessor that uses the given set of features

        pysc2 features are documented here:
        https://github.com/deepmind/pysc2/blob/master/docs/environment.md#observation

        Args:
            screen_features (list): list of pysc2.lib.features.SCREEN_FEATURES to extract from the observation
            minimap_features (list): list of pysc2.lib.features.MINIMAP_FEATURES to extract from the observation
            flat_features (list): list of structured feature name to extract from the observation
            use_gpu (bool): move computations to the GPU?
        """
        self.screen_features = screen_features
        self.minimap_features = minimap_features
        self.flat_features = flat_features
        self.use_gpu = use_gpu

        embeddings = dict()
        embeddings['screen'] = dict()
        embeddings['minimap'] = dict()

        for feature in screen_features:
            if feature.type == features.FeatureType.CATEGORICAL:
                channels = feature.scale
                out_channels = 1  # TODO: perhaps switch to log(channels) for the output
                embedding_fn = nn.Sequential(
                    nn.Conv2d(in_channels=channels, out_channels=out_channels, kernel_size=1, stride=1),
                    nn.ReLU()
                )
                if self.use_gpu:
                    embedding_fn = embedding_fn.cuda()
                embeddings['screen'][feature.name] = embedding_fn

        for feature in minimap_features:
            if feature.type == features.FeatureType.CATEGORICAL:
                channels = feature.scale
                out_channels = 1  # TODO: perhaps switch to log(channels) for the output
                embedding_fn = nn.Sequential(
                    nn.Conv2d(in_channels=channels, out_channels=out_channels, kernel_size=1, stride=1),
                    nn.ReLU()
                )
                if self.use_gpu:
                    embedding_fn = embedding_fn.cuda()
                embeddings['minimap'][feature.name] = embedding_fn
        self.embeddings = embeddings

    def process(self, observation):
        """ Preprocesses an observation from the pysc2 environment


        Categorical features are embedded into a continuous space using a one-hot encoding followed by a 1x1 convolution
        From pysc2 paper:
            "We embed all feature layers containing categorical values into a continuous space which is
            equivalent to using a one-hot encoding in the channel dimension followed by a 1 × 1 convolution."
        Continuous features are scaled logarithmically.

        Args:
            observation: observation dictionary from pysc2

        Returns:
            tuple: (minimap, screen, flat, available_actions)

            `minimap` will be a torch.autograd.Variable of shape 1 x c x w x h
            `screen` will be a torch.autograd.Variable of shape 1 x C x W x H
            `flat` will be a torch.autograd.Variable of shape 1 x F
            `available_actions` with be a Variable of size 524 containing a binary mask of available actions

        """
        # select and process desired screen features
        screen_feats = [
            self._embed_spatial_feature(feature, observation, 'screen') for feature in self.screen_features
        ]
        screen_features = torch.cat(screen_feats, dim=1)

        # select and process desired minimap features
        mm_feats = [
            self._embed_spatial_feature(feature, observation, 'minimap') for feature in self.minimap_features
        ]
        minimap_features = torch.cat(mm_feats, dim=1)

        # extract structured features
        flat_features = np.empty(0)
        for feature in self.flat_features:
            max_feature_size = constants.MAX_FLAT_SIZES[feature]
            observed = observation[feature].reshape(-1)
            feature_vector = np.pad(observed, (0, max_feature_size - len(observed)), mode='constant')
            flat_features = np.append(flat_features, feature_vector)

        flat_tensor = torch.from_numpy(flat_features).float()
        flat_tensor = flat_tensor.unsqueeze(0)  # add minibatch dimension
        if self.use_gpu:
            flat_tensor = flat_tensor.cuda()
        flat_features = Variable(flat_tensor)  # Variable with shape 1 x F

        # create mask of available actions
        TOTAL_ACTIONS = len(actions.FUNCTIONS)
        available_actions = np.zeros(TOTAL_ACTIONS, dtype=np.float32)
        available_actions[observation['available_actions']] = 1
        action_tensor = torch.from_numpy(available_actions).float()
        if self.use_gpu:
            action_tensor = action_tensor.cuda()
        available_actions = Variable(action_tensor)

        return screen_features, minimap_features, flat_features, available_actions

    def _embed_spatial_feature(self, feature, observation, feature_source):
        """ Embeds spatial features as described in the Deepmind paper

        Categorical features are embedding by one-hot encoding followed by a 1x1 convolution.
        Scalar features are scaled logarithmically.

        Args:
            feature: feature to embed
            feature_source: 'screen' for screen features; 'minimap' for minimap features

        Returns:
            torch.autograd.Variable: Variable containing embedded feature
        """
        source = observation[feature_source]
        numpy_feature = source[feature.index]
        if feature.type == features.FeatureType.SCALAR:
            # apply log transform to scalar features
            numpy_feature = np.log(numpy_feature + 1)
            numpy_feature = numpy_feature[None, :, :]  # add channel dimension ( 1 x W x H )
            tensor = torch.from_numpy(numpy_feature).float()  # create tensor
            tensor = tensor.unsqueeze(0)  # add minibatch dimension
            if self.use_gpu:
                tensor = tensor.cuda()  # move to GPU
            return Variable(tensor)  # Variable with shape 1 x 1 x W x H
        elif feature.type == features.FeatureType.CATEGORICAL:
            # apply continuous embedding to categorical features
            one_hot = onehot_encode(numpy_feature, feature.scale)  # shape: scale x W x H
            tensor = torch.from_numpy(one_hot).float()  # create tensor
            tensor = tensor.unsqueeze(0)  # add minibatch dimension
            if self.use_gpu:
                tensor = tensor.cuda()
            # lookup convolution used to embed the one-hot vector
            embedding_function = self.embeddings[feature_source][feature.name]
            # embed into continuous space
            feature_var = embedding_function(Variable(tensor))  # Variable with shape 1 x 1 x W x H
            return feature_var
        else:
            raise NotImplementedError




    def get_flat_size(self):
        """ Computes the size of the flat feature vector

        The size of the feature vector can vary based on how many units are selected, how many unit-producing buildings
        exist, how many cargo transports exist, etc.
        This function computes the length of the flat feature based on the flat features which are being extracted

        Returns:
            int: length of the flat feature vector
        """
        size = 0
        for feature in self.flat_features:
            size += constants.MAX_FLAT_SIZES[feature]
        return size


def onehot_encode(a, scale):
    """ One-hot encodes a 2d numpy array

    Source: https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy

    Args:
        a: the array to encode
        scale: the number of unique values

    Returns:
        ndarray: one-hot encoded vector with shape C x W x H

    """
    one_hot = np.zeros((a.size, scale), dtype=np.uint8)
    one_hot[np.arange(a.size), a.ravel()] = 1  # encode the feature in the third dimension
    one_hot.shape = a.shape + (scale,)  # reshape to W x H x C
    one_hot = np.transpose(one_hot, [2, 0, 1])  # reorder to C x W x H
    return one_hot
