import abc
import numpy as np


class Policy(abc.ABC):
    @abc.abstractmethod
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        Input: observation (no batch dimension)
        Output: action (no batch dimension)
        """
        raise NotImplementedError
