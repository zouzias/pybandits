# MIT License
#
# Copyright (c) 2022 Playtika Ltd.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from collections import defaultdict
from typing import Dict, List, Optional, Set, Union

from pybandits.base import (
    ActionId,
    BinaryReward,
    SmabPredictions,
    UnifiedActionId,
)
from pybandits.mab import BaseMab
from pybandits.model import BaseBeta, Beta, BetaCC, BetaMO, BetaMOCC
from pybandits.pydantic_version_compatibility import PositiveInt, field_validator, validate_call
from pybandits.quantitative_model import BaseSmabZoomingModel, SmabZoomingModel, SmabZoomingModelCC
from pybandits.strategy import (
    BestActionIdentificationBandit,
    ClassicBandit,
    CostControlBandit,
    MultiObjectiveBandit,
    MultiObjectiveCostControlBandit,
    Strategy,
)


class BaseSmabBernoulli(BaseMab):
    """
    Base model for a Stochastic Bernoulli Multi-Armed Bandit with Thompson Sampling.

    Parameters
    ----------
    actions: Dict[ActionId, Union[BaseBeta, BaseSmabZoomingModel]]
        The list of possible actions, and their associated Model.
    strategy: Strategy
        The strategy used to select actions.
    """

    actions: Dict[ActionId, Union[BaseBeta, BaseSmabZoomingModel]]

    @validate_call
    def predict(
        self,
        n_samples: PositiveInt = 1,
        forbidden_actions: Optional[Set[ActionId]] = None,
    ) -> SmabPredictions:
        """
        Predict actions.

        Parameters
        ----------
        n_samples : PositiveInt, default=1
            Number of samples to predict.
        forbidden_actions : Optional[Set[ActionId]], default=None
            Set of forbidden actions. If specified, the model will discard the forbidden_actions and it will only
            consider the remaining allowed_actions. By default, the model considers all actions as allowed_actions.
            Note that: actions = allowed_actions U forbidden_actions.

        Returns
        -------
        actions: List[UnifiedActionId]
            The actions selected by the multi-armed bandit model.
        probs: Union[List[Dict[UnifiedActionId, Probability]], List[Dict[UnifiedActionId, MOProbability]]]
            The probabilities of getting a positive reward for each action.
        """

        probs = self._get_action_probabilities(forbidden_actions=forbidden_actions, n_samples=n_samples)
        selected_actions = [self._select_epsilon_greedy_action(p=prob, actions=self.actions) for prob in probs]

        return selected_actions, probs

    @validate_call
    def _update(
        self,
        actions: List[UnifiedActionId],
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]],
        quantities: Optional[List[Union[float, List[float], None]]],
    ):
        """
        Update the stochastic Bernoulli bandit given the list of selected actions and their corresponding binary
        rewards.

        Parameters
        ----------
        actions : List[ActionId] of shape (n_samples,), e.g. ['a1', 'a2', 'a3', 'a4', 'a5']
            The selected action for each sample.
        rewards : Union[List[BinaryReward], List[List[BinaryReward]]],
            if nested list, len() should follow shape of (n_samples, n_objectives)
            The binary reward for each sample.
                If strategy is not MultiObjectiveBandit, rewards should be a list, e.g.
                    rewards = [1, 0, 1, 1, 1, ...]
                If strategy is MultiObjectiveBandit, rewards should be a list of list, e.g. (with n_objectives=2):
                    rewards = [[1, 1], [1, 0], [1, 1], [1, 0], [1, 1], ...]
        quantities : Optional[List[Union[float, List[float], None]]]
            The value associated with each action. If none, the value is not used, i.e. non-quantitative action.
        """

        rewards_dict = defaultdict(list)

        if quantities is None:
            for a, r in zip(actions, rewards):
                rewards_dict[a].append(r)
            for a in set(actions):
                self.actions[a].update(rewards=rewards_dict[a])
        else:
            quantities_dict = defaultdict(list)
            for a, v, r in zip(actions, quantities, rewards):
                if v is not None:
                    quantities_dict[a].append(v)
                rewards_dict[a].append(r)
            for a in set(actions):
                if quantities_dict[a]:  # quantitative action
                    self.actions[a].update(rewards=rewards_dict[a], quantities=quantities_dict[a])
                else:  # non-quantitative action
                    self.actions[a].update(rewards=rewards_dict[a])


class SmabBernoulli(BaseSmabBernoulli):
    """
    Stochastic Bernoulli Multi-Armed Bandit with Thompson Sampling.

    Reference: Analysis of Thompson Sampling for the Multi-armed Bandit Problem (Agrawal and Goyal, 2012)
               http://proceedings.mlr.press/v23/agrawal12/agrawal12.pdf

    Parameters
    ----------
    actions: Dict[UnifiedActionId, Union[Beta, SmabZoomingModel]]
        The list of possible actions, and their associated Model.
    strategy: ClassicBandit
        The strategy used to select actions.
    """

    actions: Dict[ActionId, Union[Beta, SmabZoomingModel]]
    strategy: ClassicBandit


class SmabBernoulliBAI(BaseSmabBernoulli):
    """
    Stochastic Bernoulli Multi-Armed Bandit with Thompson Sampling, and Best Action Identification strategy.

    Reference: Analysis of Thompson Sampling for the Multi-armed Bandit Problem (Agrawal and Goyal, 2012)
               http://proceedings.mlr.press/v23/agrawal12/agrawal12.pdf

    Parameters
    ----------
    actions: Dict[ActionId, Union[Beta, SmabZoomingModel]]
        The list of possible actions, and their associated Model.
    strategy: BestActionIdentificationBandit
        The strategy used to select actions.
    """

    actions: Dict[ActionId, Union[Beta, SmabZoomingModel]]
    strategy: BestActionIdentificationBandit


class SmabBernoulliCC(BaseSmabBernoulli):
    """
    Stochastic Bernoulli Multi-Armed Bandit with Thompson Sampling, and Cost Control strategy.

    The sMAB is extended to include a control of the action cost. Each action is associated with a predefined "cost".
    At prediction time, the model considers the actions whose expected rewards is above a pre-defined lower bound. Among
    these actions, the one with the lowest associated cost is recommended. The expected reward interval for feasible
    actions is defined as [(1-subsidy_factor) * max_p, max_p], where max_p is the highest expected reward sampled value.

    Reference: Thompson Sampling for Contextual Bandit Problems with Auxiliary Safety Constraints (Daulton et al., 2019)
               https://arxiv.org/abs/1911.00638

               Multi-Armed Bandits with Cost Subsidy (Sinha et al., 2021)
               https://arxiv.org/abs/2011.01488

    Parameters
    ----------
    actions: Dict[ActionId, Union[BetaCC, SmabZoomingModelCC]]
        The list of possible actions, and their associated Model.
    strategy: CostControlBandit
        The strategy used to select actions.
    """

    actions: Dict[ActionId, Union[BetaCC, SmabZoomingModelCC]]
    strategy: CostControlBandit


class BaseSmabBernoulliMO(BaseSmabBernoulli):
    """
    Base model for a Stochastic Bernoulli Multi-Armed Bandit with Thompson Sampling implementation, and Multi-Objectives
    strategy.

    Parameters
    ----------
    actions: Dict[ActionId, BetaMO]
        The list of possible actions, and their associated Model.
    strategy: Strategy
        The strategy used to select actions.
    """

    actions: Dict[ActionId, BetaMO]
    strategy: Strategy

    @field_validator("actions", mode="after")
    @classmethod
    def all_actions_have_same_number_of_objectives(cls, actions: Dict[ActionId, BetaMO]):
        n_objs_per_action = [len(beta.models) for beta in actions.values()]
        if len(set(n_objs_per_action)) != 1:
            raise ValueError("All actions should have the same number of objectives")
        return actions


class SmabBernoulliMO(BaseSmabBernoulliMO):
    """
    Stochastic Bernoulli Multi-Armed Bandit with Thompson Sampling, and Multi-Objectives strategy.

    The reward pertaining to an action is a multidimensional vector instead of a scalar value. In this setting,
    different actions are compared according to Pareto order between their expected reward vectors, and those actions
    whose expected rewards are not inferior to that of any other actions are called Pareto optimal actions, all of which
    constitute the Pareto front.

    Reference: Thompson Sampling for Multi-Objective Multi-Armed Bandits Problem (Yahyaa and Manderick, 2015)
               https://www.researchgate.net/publication/272823659_Thompson_Sampling_for_Multi-Objective_Multi-Armed_Bandits_Problem

    Parameters
    ----------
    actions: Dict[ActionId, BetaMO]
        The list of possible actions, and their associated Model.
    strategy: MultiObjectiveBandit
        The strategy used to select actions.
    """

    actions: Dict[ActionId, BetaMO]
    strategy: MultiObjectiveBandit


class SmabBernoulliMOCC(BaseSmabBernoulliMO):
    """
    Stochastic Bernoulli Multi-Armed Bandit with Thompson Sampling implementation for Multi-Objective (MO) with Cost
    Control (CC) strategy.

    This Bandit allows the reward to be a multidimensional vector and include a control of the action cost. It merges
    the Multi-Objective and Cost Control strategies.

    Parameters
    ----------
    actions: Dict[ActionId, BetaMOCC]
        The list of possible actions, and their associated Model.
    strategy: MultiObjectiveCostControlBandit
        The strategy used to select actions.
    """

    actions: Dict[ActionId, BetaMOCC]
    strategy: MultiObjectiveCostControlBandit
