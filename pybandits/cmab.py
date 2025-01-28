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

from numpy import array
from numpy.typing import ArrayLike

from pybandits.base import (
    ActionId,
    BinaryReward,
    CmabPredictions,
    UnifiedActionId,
)
from pybandits.mab import BaseMab
from pybandits.model import BaseBayesianLogisticRegression, BayesianLogisticRegression, BayesianLogisticRegressionCC
from pybandits.pydantic_version_compatibility import field_validator, validate_call
from pybandits.quantitative_model import BaseCmabZoomingModel, CmabZoomingModel, CmabZoomingModelCC
from pybandits.strategy import (
    BestActionIdentificationBandit,
    ClassicBandit,
    CostControlBandit,
)


class BaseCmabBernoulli(BaseMab):
    """
    Base model for a Contextual Multi-Armed Bandit for Bernoulli bandits with Thompson Sampling.

    Parameters
    ----------
    actions : Dict[ActionId, Union[BaseBayesianLogisticRegression, BaseCmabZoomingModel]]
        The list of possible actions, and their associated Model.
    strategy : Strategy
        The strategy used to select actions.
    """

    actions: Dict[ActionId, Union[BaseBayesianLogisticRegression, BaseCmabZoomingModel]]
    _predict_with_proba: bool

    @staticmethod
    def _maybe_crawl_model(model: Union[BaseBayesianLogisticRegression, BaseCmabZoomingModel]):
        return list(model.sub_actions.values())[0] if isinstance(model, BaseCmabZoomingModel) else model

    @field_validator("actions", mode="after")
    @classmethod
    def check_models(cls, v):
        action_models = list(v.values())
        first_action = action_models[0]
        test_first_action = cls._maybe_crawl_model(first_action)
        for action in action_models[1:]:
            test_action = cls._maybe_crawl_model(action)
            if not len(test_action.betas) == len(test_first_action.betas):
                raise AttributeError("All actions should have the same number of betas.")
            if not test_action.update_method == test_first_action.update_method:
                raise AttributeError("All actions should have the same update method.")
            if not test_action.update_kwargs == test_first_action.update_kwargs:
                raise AttributeError("All actions should have the same update kwargs.")
        return v

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def predict(
        self,
        context: ArrayLike,
        forbidden_actions: Optional[Set[ActionId]] = None,
    ) -> CmabPredictions:
        """
        Predict actions.

        Parameters
        ----------
        context: ArrayLike of shape (n_samples, n_features)
            Matrix of contextual features.
        forbidden_actions : Optional[Set[ActionId]], default=None
            Set of forbidden actions. If specified, the model will discard the forbidden_actions and it will only
            consider the remaining allowed_actions. By default, the model considers all actions as allowed_actions.
            Note that: actions = allowed_actions U forbidden_actions.

        Returns
        -------
        actions: List[ActionId]
            The actions selected by the multi-armed bandit model.
        probs: Union[List[Dict[UnifiedActionId, Probability]], List[Dict[UnifiedActionId, MOProbability]]]
            The probabilities of getting a positive reward for each action.
        ws : Union[List[Dict[UnifiedActionId, float]], List[Dict[UnifiedActionId, List[float]]]]
            The weighted sum of logistic regression logits.
        """

        # cast inputs to numpy arrays to facilitate their manipulation
        context = array(context)

        if len(context) < 1:
            raise AttributeError("Context must have at least one row")

        # p is a dict of the sampled probability "prob" and weighted_sum "ws", e.g.
        #
        # p = {'a1': ([0.5, 0.2, 0.3], [200, 100, 130]), 'a2': ([0.4, 0.5, 0.6], [180, 200, 230]), ...}
        #               |               |                           |               |
        #              prob             ws                          prob            ws
        probs_weights = self._get_action_probabilities(forbidden_actions=forbidden_actions, context=context)

        probs = [
            {a: x[0] for a, x in prob_weight.items()} for prob_weight in probs_weights
        ]  # e.g. prob = {'a1': [0.5, 0.4, ...], 'a2': [0.4, 0.3, ...], ...}
        weighted_sums = [
            {a: x[1] for a, x in prob_weight.items()} for prob_weight in probs_weights
        ]  # e.g. ws = {'a1': [200, 100, ...], 'a2': [100, 50, ...], ...}

        # select either "prob" or "ws" to use as input argument in select_actions()
        p_to_select_action = probs if self._predict_with_proba else weighted_sums

        # predict actions, probs, weighted_sums
        selected_actions = [self._select_epsilon_greedy_action(p=p, actions=self.actions) for p in p_to_select_action]

        return selected_actions, probs, weighted_sums

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def _update(
        self,
        actions: List[UnifiedActionId],
        rewards: List[Union[BinaryReward, List[BinaryReward]]],
        quantities: Optional[List[Union[float, List[float], None]]],
        context: ArrayLike,
    ):
        """
        Update the contextual Bernoulli bandit given the list of selected actions and their corresponding binary
        rewards.

        Parameters
        ----------

        actions : List[UnifiedActionId] of shape (n_samples,), e.g. ['a1', 'a2', 'a3', 'a4', 'a5']
            The selected action for each sample.
        rewards : List[Union[BinaryReward, List[BinaryReward]]] of shape (n_samples, n_objectives)
            The binary reward for each sample.
                If strategy is not MultiObjectiveBandit, rewards should be a list, e.g.
                    rewards = [1, 0, 1, 1, 1, ...]
                If strategy is MultiObjectiveBandit, rewards should be a list of list, e.g. (with n_objectives=2):
                    rewards = [[1, 1], [1, 0], [1, 1], [1, 0], [1, 1], ...]
        quantities : Optional[List[Union[float, List[float], None]]]
            The value associated with each action. If none, the value is not used, i.e. non-quantitative action.
        context: ArrayLike of shape (n_samples, n_features)
            Matrix of contextual features.
        """
        context = array(context)  # cast inputs to numpy arrays to facilitate their manipulation

        rewards_dict = defaultdict(list)

        if quantities is None:
            for a, r in zip(actions, rewards):
                rewards_dict[a].append(r)
            for a in set(actions):
                mask = [action == a for action in actions]
                self.actions[a].update(context=context[mask], rewards=rewards_dict[a])
        else:
            quantities_dict = defaultdict(list)
            for a, v, r in zip(actions, quantities, rewards):
                if v is not None:
                    quantities_dict[a].append(v)
                rewards_dict[a].append(r)
            for a in set(actions):
                mask = [action == a for action in actions]
                if quantities_dict[a]:  # quantitative action
                    self.actions[a].update(
                        context=context[mask], rewards=rewards_dict[a], quantities=quantities_dict[a]
                    )
                else:  # non-quantitative action
                    self.actions[a].update(context=context[mask], rewards=rewards_dict[a])


class CmabBernoulli(BaseCmabBernoulli):
    """
    Contextual Bernoulli Multi-Armed Bandit with Thompson Sampling.

    Reference: Thompson Sampling for Contextual Bandits with Linear Payoffs (Agrawal and Goyal, 2014)
               https://arxiv.org/pdf/1209.3352.pdf

    Parameters
    ----------
    actions: Dict[ActionId, Union[BayesianLogisticRegression, CmabZoomingModel]]
        The list of possible actions, and their associated Model.
    strategy: ClassicBandit
        The strategy used to select actions.
    """

    actions: Dict[ActionId, Union[BayesianLogisticRegression, CmabZoomingModel]]
    strategy: ClassicBandit
    _predict_with_proba: bool = False


class CmabBernoulliBAI(BaseCmabBernoulli):
    """
    Contextual Bernoulli Multi-Armed Bandit with Thompson Sampling, and Best Action Identification strategy.

    Reference: Analysis of Thompson Sampling for the Multi-armed Bandit Problem (Agrawal and Goyal, 2012)
               http://proceedings.mlr.press/v23/agrawal12/agrawal12.pdf

    Parameters
    ----------
    actions: Dict[ActionId, Union[BayesianLogisticRegression, CmabZoomingModel]]
        The list of possible actions, and their associated Model.
    strategy: BestActionIdentificationBandit
        The strategy used to select actions.
    """

    actions: Dict[ActionId, Union[BayesianLogisticRegression, CmabZoomingModel]]
    strategy: BestActionIdentificationBandit
    _predict_with_proba: bool = False


class CmabBernoulliCC(BaseCmabBernoulli):
    """
    Contextual Bernoulli Multi-Armed Bandit with Thompson Sampling, and Cost Control strategy.

    The Cmab is extended to include a control of the action cost. Each action is associated with a predefined "cost".
    At prediction time, the model considers the actions whose expected rewards is above a pre-defined lower bound. Among
    these actions, the one with the lowest associated cost is recommended. The expected reward interval for feasible
    actions is defined as [(1-subsidy_factor) * max_p, max_p], where max_p is the highest expected reward sampled value.

    Reference: Thompson Sampling for Contextual Bandit Problems with Auxiliary Safety Constraints (Daulton et al., 2019)
               https://arxiv.org/abs/1911.00638

               Multi-Armed Bandits with Cost Subsidy (Sinha et al., 2021)
               https://arxiv.org/abs/2011.01488

    Parameters
    ----------
    actions: Dict[ActionId, Union[BayesianLogisticRegressionCC, CmabZoomingModelCC]]
        The list of possible actions, and their associated Model.
    strategy: CostControlBandit
        The strategy used to select actions.
    """

    actions: Dict[ActionId, Union[BayesianLogisticRegressionCC, CmabZoomingModelCC]]
    strategy: CostControlBandit
    _predict_with_proba: bool = True
