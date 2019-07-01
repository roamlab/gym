import gym
import numpy as np
from collections import OrderedDict


__all__ = ['FlattenDictWrapper', 'DictInputWrapper']


class FlattenDictWrapper(gym.ObservationWrapper):
    """Flattens selected keys of a Dict observation space into
    an array.
    """
    def __init__(self, env, dict_keys):
        super(FlattenDictWrapper, self).__init__(env)
        self.dict_keys = dict_keys

        # Figure out observation_space dimension.
        size = 0
        for key in dict_keys:
            shape = self.env.observation_space.spaces[key].shape
            size += np.prod(shape)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype='float32')

    def observation(self, observation):
        assert isinstance(observation, dict)
        obs = []
        for key in self.dict_keys:
            obs.append(observation[key].ravel())
        return np.concatenate(obs)


class DictInputWrapper(gym.ObservationWrapper):
    """ Wrapper to enable use of Dict observation space as input to policy and algorithm """

    def __init__(self, env, dict_keys):
        super(DictInputWrapper, self).__init__(env)
        self.dict_keys = dict_keys

        # Create a new Dict space, add shape and dtype attributes
        spaces = OrderedDict()
        size = 0
        dtype = None
        for key in dict_keys:
            space = self.env.observation_space.spaces[key]
            shape = space.shape
            spaces[key] = space
            size += np.prod(shape)
            if dtype:
                assert dtype == space.dtype
            else:
                dtype = space.dtype
        self.observation_space = gym.spaces.Dict(spaces)
        self.observation_space.shape = (size, )
        self.observation_space.dtype = dtype

    def observation(self, observation):
        assert isinstance(observation, dict)
        obs = {}
        for key in self.dict_keys:
            obs[key] = observation[key]
        return obs
