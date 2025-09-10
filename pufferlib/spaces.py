import numpy as np
import gymnasium

try:
    import gym  # Optional; used only for isinstance checks
    _GYM_OK = True
except Exception:
    gym = None
    _GYM_OK = False

if _GYM_OK:
    Box = (gym.spaces.Box, gymnasium.spaces.Box)
    Dict = (gym.spaces.Dict, gymnasium.spaces.Dict)
    Discrete = (gym.spaces.Discrete, gymnasium.spaces.Discrete)
    MultiBinary = (gym.spaces.MultiBinary, gymnasium.spaces.MultiBinary)
    MultiDiscrete = (gym.spaces.MultiDiscrete, gymnasium.spaces.MultiDiscrete)
    Tuple = (gym.spaces.Tuple, gymnasium.spaces.Tuple)
else:
    Box = (gymnasium.spaces.Box,)
    Dict = (gymnasium.spaces.Dict,)
    Discrete = (gymnasium.spaces.Discrete,)
    MultiBinary = (gymnasium.spaces.MultiBinary,)
    MultiDiscrete = (gymnasium.spaces.MultiDiscrete,)
    Tuple = (gymnasium.spaces.Tuple,)

def joint_space(space, n):
    if isinstance(space, Discrete):
        return gymnasium.spaces.MultiDiscrete([space.n] * n)
    elif isinstance(space, MultiDiscrete):
        return gymnasium.spaces.Box(low=0,
            high=np.repeat(space.nvec[None] - 1, n, axis=0),
            shape=(n, len(space)), dtype=space.dtype)
    elif isinstance(space, Box):
        low = np.repeat(space.low[None], n, axis=0)
        high = np.repeat(space.high[None], n, axis=0)
        return gymnasium.spaces.Box(low=low, high=high,
            shape=(n, *space.shape), dtype=space.dtype)
    else:
        raise ValueError(f'Unsupported space: {space}')
