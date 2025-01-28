from abc import ABC, abstractmethod
from typing import Callable, List, Union

import numpy as np

from pybandits.base import (
    BinaryReward,
    MOProbability,
    Probability,
    ProbabilityWeight,
    PyBanditsBaseModel,
    QuantitativeMOProbability,
    QuantitativeProbability,
    QuantitativeProbabilityWeight,
)
from pybandits.pydantic_version_compatibility import NonNegativeFloat


class BaseModel(PyBanditsBaseModel, ABC):
    """
    Class to model the prior distributions of standard actions and quantitative actions.
    """

    @abstractmethod
    def sample_proba(
        self, **kwargs
    ) -> Union[
        List[Probability],
        List[MOProbability],
        List[ProbabilityWeight],
        List[QuantitativeProbability],
        List[QuantitativeMOProbability],
        List[QuantitativeProbabilityWeight],
    ]:
        """
        Sample the probability of getting a positive reward.
        """

    @abstractmethod
    def update(self, rewards: Union[List[BinaryReward], List[List[BinaryReward]]], **kwargs):
        """
        Update the model parameters.

        Parameters
        ----------
        rewards : Union[List[BinaryReward], List[List[BinaryReward]]],
            if nested list, len() should follow shape of (n_samples, n_objectives)
            The binary reward for each sample.
                If strategy is not MultiObjectiveBandit, rewards should be a list, e.g.
                    rewards = [1, 0, 1, 1, 1, ...]
                If strategy is MultiObjectiveBandit, rewards should be a list of list, e.g. (with n_objectives=2):
                    rewards = [[1, 1], [1, 0], [1, 1], [1, 0], [1, 1], ...]
        """


class BaseModelSO(BaseModel, ABC):
    """
    Class to model the prior distributions of standard actions and quantitative actions for single objective.
    """

    @abstractmethod
    def sample_proba(
        self, **kwargs
    ) -> Union[
        List[Probability], List[ProbabilityWeight], List[QuantitativeProbability], List[QuantitativeProbabilityWeight]
    ]:
        """
        Sample the probability of getting a positive reward.
        """

    @abstractmethod
    def update(self, rewards: List[BinaryReward], **kwargs):
        """
        Update the model parameters.

        Parameters
        ----------
        rewards : List[BinaryReward],
            The binary reward for each sample.
        """


class BaseModelMO(BaseModel, ABC):
    """
    Class to model the prior distributions of standard actions and quantitative actions for multi-objective.

    Parameters
    ----------
    models : List[BaseModelSO]
        The list of models for each objective.
    """

    models: List[BaseModelSO]

    @abstractmethod
    def sample_proba(self, **kwargs) -> Union[List[MOProbability], List[QuantitativeMOProbability]]:
        """
        Sample the probability of getting a positive reward.
        """

    @abstractmethod
    def update(self, rewards: List[List[BinaryReward]], **kwargs):
        """
        Update the model parameters.

        Parameters
        ----------
        rewards : List[List[BinaryReward]],
            if nested list, len() should follow shape of (n_samples, n_objectives)
            The binary rewards for each sample.
                If strategy is not MultiObjectiveBandit, rewards should be a list, e.g.
                    rewards = [1, 0, 1, 1, 1, ...]
                If strategy is MultiObjectiveBandit, rewards should be a list of list, e.g. (with n_objectives=2):
                    rewards = [[1, 1], [1, 0], [1, 1], [1, 0], [1, 1], ...]
        """


class BaseModelCC(PyBanditsBaseModel, ABC):
    """
    Class to model action cost.

    Parameters
    ----------
    cost: Union[NonNegativeFloat, Callable[[Union[float, NonNegativeFloat]], NonNegativeFloat]]
        Cost associated to the Beta distribution.
    """

    cost: Union[NonNegativeFloat, Callable[[Union[float, np.ndarray]], NonNegativeFloat]]
