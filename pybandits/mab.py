# MIT License
#
# Copyright (c) 2023 Playtika Ltd.
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

import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, get_args, get_origin

import numpy as np

from pybandits.base import (
    ACTION_IDS_PREFIX,
    QUANTITATIVE_ACTION_IDS_PREFIX,
    ActionId,
    ActionRewardLikelihood,
    BinaryReward,
    Float01,
    MOProbability,
    MOProbabilityWeight,
    Predictions,
    Probability,
    ProbabilityWeight,
    PyBanditsBaseModel,
    Serializable,
    UnifiedActionId,
)
from pybandits.base_model import BaseModel
from pybandits.model import Model, ModelMO
from pybandits.pydantic_version_compatibility import (
    field_validator,
    validate_call,
)
from pybandits.quantitative_model import QuantitativeModel
from pybandits.strategy import Strategy
from pybandits.utils import extract_argument_names


class BaseMab(PyBanditsBaseModel, ABC):
    """
    Multi-armed bandit superclass.

    Parameters
    ----------
    actions : Dict[ActionId, Model]
        The list of possible actions, and their associated Model.
    strategy : Strategy
        The strategy used to select actions.
    epsilon : Optional[Float01], 0 if not specified.
        The probability of selecting a random action.
    default_action : Optional[ActionId], None if not specified.
        The default action to select with a probability of epsilon when using the epsilon-greedy approach.
        If `default_action` is None, a random action from the action set will be selected with a probability of epsilon.
    strategy_kwargs : Dict[str, Any]
        Relevant only if strategy was not provided. This argument contains the parameters for the strategy,
        which in turn will be used to instantiate the strategy.
    """

    actions: Dict[ActionId, BaseModel]
    strategy: Strategy
    epsilon: Optional[Float01] = None
    default_action: Optional[UnifiedActionId] = None

    def __init__(
        self,
        actions: Dict[ActionId, BaseModel],
        epsilon: Optional[Float01] = None,
        default_action: Optional[ActionId] = None,
        **strategy_kwargs,
    ):
        if "strategy" in strategy_kwargs:
            strategy = strategy_kwargs["strategy"]
            if len(strategy_kwargs) > 1:
                raise ValueError("strategy should be the only keyword argument.")
        else:
            strategy_class = self.model_fields["strategy"].annotation
            strategy = strategy_class(**strategy_kwargs)

        super().__init__(actions=actions, strategy=strategy, epsilon=epsilon, default_action=default_action)

    ############################################ Instance Input Validators #############################################

    @field_validator("actions", mode="before")
    @classmethod
    def at_least_one_action_is_defined(cls, v):
        # validate number of actions
        if len(v) == 0:
            raise AttributeError("At least one action should be defined.")
        elif len(v) == 1:
            warnings.warn("Only a single action was supplied. This MAB will be deterministic.")
        return v

    def model_post_init(self, __context: Any) -> None:
        if not self.epsilon and self.default_action:
            raise AttributeError("A default action should only be defined when epsilon is defined.")
        if self.default_action and self.default_action not in self.actions:
            raise AttributeError("The default action must be valid action defined in the actions set.")
        if (
            self.default_action
            and isinstance(self.default_action, tuple)
            and not isinstance(self.actions[self.default_action[0]], QuantitativeModel)
        ):
            raise AttributeError("Quantitative default action requires a quantitative action model.")
        if (
            self.default_action
            and isinstance(self.default_action, str)
            and not isinstance(self.actions[self.default_action], (Model, ModelMO))
        ):
            raise AttributeError("Standard default action requires a standard action model.")

    ############################################# Method Input Validators ##############################################

    def _get_valid_actions(self, forbidden_actions: Optional[Set[ActionId]]) -> Set[ActionId]:
        """
        Given a set of forbidden action IDs, return a set of valid action IDs.

        Parameters
        ----------
        forbidden_actions: Optional[Set[ActionId]]
            The set of forbidden action IDs.

        Returns
        -------
        valid_actions: Set[ActionId]
            The list of valid (i.e. not forbidden) action IDs.
        """
        if forbidden_actions is None:
            forbidden_actions = set()

        if not all(a in self.actions.keys() for a in forbidden_actions):
            raise ValueError("forbidden_actions contains invalid action IDs.")
        valid_actions = set(self.actions.keys()) - forbidden_actions
        if len(valid_actions) == 0:
            raise ValueError("All actions are forbidden. You must allow at least 1 action.")
        if self.default_action and self.default_action not in valid_actions:
            raise ValueError("The default action is forbidden.")

        return valid_actions

    ####################################################################################################################

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def update(
        self,
        actions: List[ActionId],
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]],
        quantities: Optional[List[Union[float, List[float], None]]] = None,
        **kwargs,
    ):
        """
        Update the multi-armed bandit model.

        actions: List[ActionId]
            The selected action for each sample.
        rewards: List[Union[BinaryReward, List[BinaryReward]]]
            The reward for each sample.
        quantities: Optional[List[Union[float, List[float], None]]]
            The value associated with each action. If none, the value is not used, i.e. non-quantitative action.
        context: Optional[ArrayLike]
            The context for each sample.
        """
        invalid = set(actions) - set(self.actions.keys())
        if invalid:
            raise AttributeError(f"The following invalid action(s) were specified: {invalid}.")
        self._validate_params_lengths(actions=actions, rewards=rewards, quantities=quantities, **kwargs)
        if quantities is None:
            if not all(isinstance(self.actions[action], (Model, ModelMO)) for action in actions):
                raise ValueError("Quantitative actions require defined quantities.")
        else:
            if not all(
                q is not None for a, q in zip(actions, quantities) if isinstance(self.actions[a], QuantitativeModel)
            ):
                raise ValueError("Quantitative actions require defined quantities.")
            if not all(q is None for a, q in zip(actions, quantities) if isinstance(self.actions[a], (Model, ModelMO))):
                raise ValueError("Standard actions should not have defined quantities.")
        self._update(actions, rewards, quantities, **kwargs)

    @abstractmethod
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def _update(
        self,
        actions: List[ActionId],
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]],
        quantities: Optional[List[Union[float, List[float], None]]],
        **kwargs,
    ):
        """
        Update the multi-armed bandit model.

        actions: List[ActionId]
            The selected action for each sample.
        rewards: List[Union[BinaryReward, List[BinaryReward]]]
            The reward for each sample.
        quantities: Optional[List[Union[float, List[float], None]]]
            The value associated with each action. If none, the value is not used, i.e. non-quantitative action.
        """

    @staticmethod
    def _transform_nested_list(lst: List[List[Dict]]):
        return [{k: v for d in single_action_dicts for k, v in d.items()} for single_action_dicts in zip(*lst)]

    @staticmethod
    def _is_so_standard_action(value: Any) -> bool:
        #       Probability                                      ProbabilityWeight
        return isinstance(value, float) or (isinstance(value, tuple) and isinstance(value[0], float))

    @staticmethod
    def _is_so_quantitative_action(value: Any) -> bool:
        return isinstance(value, tuple) and isinstance(value[0], tuple)

    @classmethod
    def _is_standard_action(cls, value: Any) -> bool:
        return cls._is_so_standard_action(value) or (isinstance(value, list) and cls._is_so_standard_action(value[0]))

    @classmethod
    def _is_quantitative_action(cls, value: Any) -> bool:
        return cls._is_so_quantitative_action(value) or (
            isinstance(value, list) and cls._is_so_quantitative_action(value[0])
        )

    def _get_action_probabilities(
        self, forbidden_actions: Optional[Set[ActionId]] = None, **kwargs
    ) -> Union[
        List[Dict[UnifiedActionId, Probability]],
        List[Dict[UnifiedActionId, ProbabilityWeight]],
        List[Dict[UnifiedActionId, MOProbability]],
        List[Dict[UnifiedActionId, MOProbabilityWeight]],
    ]:
        """
        Get the probability of getting a positive reward for each action.

        Parameters
        ----------
        forbidden_actions : Optional[Set[ActionId]], default=None
            Set of forbidden actions. If specified, the model will discard the forbidden_actions and it will only
            consider the remaining allowed_actions. By default, the model considers all actions as allowed_actions.
            Note that: actions = allowed_actions U forbidden_actions.

        Returns
        -------
        action_probabilities: Union[List[Dict[UnifiedActionId, Probability]], List[Dict[UnifiedActionId, ProbabilityWeight]], List[Dict[UnifiedActionId, MOProbability]], List[Dict[UnifiedActionId, MOProbabilityWeight]]]
            The probability of getting a positive reward for each action.
        """

        valid_actions = self._get_valid_actions(forbidden_actions)
        action_probabilities = {
            action: model.sample_proba(**kwargs) for action, model in self.actions.items() if action in valid_actions
        }
        # Handle standard actions for which the value is a (probability, weight) tuple
        list_transformations = [
            [{key: proba} for proba in value]
            for key, value in action_probabilities.items()
            if self._is_standard_action(value[0])
        ]
        list_transformations = self._transform_nested_list(list_transformations)
        # Handle quantitative actions, for which the value is a tuple of
        # tuples of (quantity, (probability, weight) or probability)
        tuple_transformations = [
            [{(key, quantity): proba for quantity, proba in sample} for sample in value]
            for key, value in action_probabilities.items()
            if self._is_quantitative_action(value[0])
        ]
        tuple_transformations = self._transform_nested_list(tuple_transformations)
        if not list_transformations and not tuple_transformations:
            return []
        if not list_transformations:  # No standard actions
            list_transformations = [dict() for _ in range(len(tuple_transformations))]
        if not tuple_transformations:  # No quantitative actions
            tuple_transformations = [dict() for _ in range(len(list_transformations))]
        if not len(list_transformations) == len(tuple_transformations):
            raise ValueError("The number of standard and quantitative actions should be the same.")
        action_probabilities = [
            {**list_dict, **dict_dict} for list_dict, dict_dict in zip(list_transformations, tuple_transformations)
        ]
        return action_probabilities

    @abstractmethod
    @validate_call
    def predict(self, forbidden_actions: Optional[Set[ActionId]] = None, **kwargs) -> Predictions:
        """
        Predict actions.

        Parameters
        ----------
        forbidden_actions : Optional[Set[ActionId]], default=None
            Set of forbidden actions. If specified, the model will discard the forbidden_actions and it will only
            consider the remaining allowed_actions. By default, the model considers all actions as allowed_actions.
            Note that: actions = allowed_actions U forbidden_actions.

        Returns
        -------
        actions: List[ActionId] of shape (n_samples,)
            The actions selected by the multi-armed bandit model.
        probs: List[Dict[ActionId, Probability]] of shape (n_samples,)
            The probabilities of getting a positive reward for each action
        ws : List[Dict[ActionId, float]], only relevant for some of the MABs
            The weighted sum of logistic regression logits..
        """

    def get_state(self) -> (str, dict):
        """
        Access the complete model internal state, enough to create an exact copy of the same model from it.
        Returns
        -------
        model_class_name: str
            The name of the class of the model.
        model_state: dict
            The internal state of the model (actions, scores, etc.).
        """
        model_name = self.__class__.__name__
        state: dict = self._apply_version_adjusted_method("model_dump", "dict")
        return model_name, state

    @validate_call
    def _select_epsilon_greedy_action(
        self,
        p: ActionRewardLikelihood,
        actions: Optional[Dict[ActionId, BaseModel]] = None,
    ) -> ActionId:
        """
        Wraps self.strategy.select_action function with epsilon-greedy strategy,
        such that with probability epsilon a default_action is selected,
        and with probability 1-epsilon the select_action function is triggered to choose action.
        If no default_action is provided, a random action is selected.

        Reference: Reinforcement Learning: An Introduction, Ch. 2 (Sutton and Burto, 2018)
               https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf&ved=2ahUKEwjMy8WV9N2HAxVe0gIHHVjjG5sQFnoECEMQAQ&usg=AOvVaw3bKK-Y_1kf6XQVwR-UYrBY

        Parameters
        ----------
        p: Union[Dict[ActionId, float], Dict[ActionId, Probability], Dict[ActionId, List[Probability]]]
            The dictionary or actions and their sampled probability of getting a positive reward.
            For MO strategy, the sampled probability is a list with elements corresponding to the objectives.
        actions: Optional[Dict[ActionId, Model]]
            The dictionary of actions and their associated Model.

        Returns
        -------
        selected_action: ActionId
            The selected action.

        Raises
        ------
        KeyError
            If self.default_action is not present as a key in the probabilities dictionary.
        """

        if self.epsilon:
            if self.default_action and self.default_action not in p.keys():
                raise KeyError(f"Default action {self.default_action} not in actions.")
            if np.random.binomial(1, self.epsilon):
                if self.default_action:
                    selected_action = self.default_action
                else:
                    selected_action = np.random.choice(list(set(a[0] if isinstance(a, tuple) else a for a in p.keys())))
                    if isinstance(self.actions[selected_action], QuantitativeModel):
                        selected_action = (
                            selected_action,
                            tuple(np.random.random(self.actions[selected_action].dimension)),
                        )
            else:
                selected_action = self.strategy.select_action(p=p, actions=actions)
        else:
            selected_action = self.strategy.select_action(p=p, actions=actions)
        return selected_action

    @classmethod
    def from_state(cls, state: Dict[str, Serializable]) -> "BaseMab":
        """
        Create a new instance of the class from a given model state.
        The state can be obtained by applying get_state() to a model.

        Parameters
        ----------
        state: dict
            The internal state of a model (actions, strategy, etc.) of the same type.

        Returns
        -------
        model: BaseMab
            The new model instance.

        """
        return cls.model_validate(state)

    @classmethod
    def cold_start(
        cls,
        action_ids: Optional[Set[ActionId]] = None,
        quantitative_action_ids: Optional[Set[ActionId]] = None,
        epsilon: Optional[Float01] = None,
        default_action: Optional[ActionId] = None,
        **kwargs,
    ) -> "BaseMab":
        """
        Factory method to create a Multi-Armed Bandit with Thompson Sampling, with default
        parameters.

        Parameters
        ----------
        action_ids : Optional[Set[ActionId]]
            The list of possible actions.
        quantitative_action_ids : Optional[Set[ActionId]]
            The list of quantitative actions.
        epsilon : Optional[Float01]
            epsilon for epsilon-greedy approach. If None, epsilon-greedy is not used.
        default_action : Optional[ActionId]
            The default action to select with a probability of epsilon when using the epsilon-greedy approach.
            If `default_action` is None, a random action from the action set will be selected with a probability of epsilon.
        kwargs : Dict[str, Any]
            Additional parameters for the mab and for the action model.

        Returns
        -------
        mab: BaseMab
            Multi-Armed Bandit
        """
        action_specific_kwargs, quantitative_action_specific_kwargs, kwargs = cls._extract_action_specific_kwargs(
            **kwargs
        )

        # Extract inner_action_ids
        inner_action_ids = action_ids or set(action_specific_kwargs)
        inner_quantitative_action_ids = quantitative_action_ids or set(quantitative_action_specific_kwargs)
        if not inner_action_ids and not inner_quantitative_action_ids:
            raise ValueError("At least one action should be defined.")

        # Assign model for each action
        (
            model_cold_start,
            quantitative_model_cold_start,
            action_general_kwargs,
            quantitative_action_general_kwargs,
        ) = cls._extract_action_model_class_and_attributes(kwargs)

        # Instantiate the actions
        all_actions = {}
        for action_ids, cold_start, general_kwargs, specific_kwargs in zip(
            [inner_action_ids, inner_quantitative_action_ids],
            [model_cold_start, quantitative_model_cold_start],
            [action_general_kwargs, quantitative_action_general_kwargs],
            [action_specific_kwargs, quantitative_action_specific_kwargs],
        ):
            for a in action_ids:
                all_actions[a] = cold_start(**general_kwargs, **specific_kwargs.get(a, {}))

        # Instantiate the MAB
        strategy_class = cls.model_fields["strategy"].annotation
        strategy = strategy_class(**kwargs)
        mab = cls(actions=all_actions, strategy=strategy, epsilon=epsilon, default_action=default_action)
        return mab

    @staticmethod
    def _extract_action_specific_kwargs(**kwargs) -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, Any]]:
        """
        Utility function to extract kwargs that are specific for each action when constructing the action model.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            Additional parameters for the mab and for the action model.

        Returns
        -------
        action_specific_kwargs : Dict[str, Dict]
            Dictionary of actions and the parameters of their associated model.
        quantitative_action_specific_kwargs : Dict[str, Dict]
            Dictionary of quantitative actions and the parameters of their associated model.
        kwargs : Dict[str, Any]
            Dictionary of parameters and their quantities, without the action_specific_kwargs.
        """
        action_specific_kwargs = defaultdict(dict)
        quantitative_action_specific_kwargs = defaultdict(dict)
        for keyword in list(kwargs):
            argument = kwargs[keyword]
            for prefix, target_kwargs in zip(
                [ACTION_IDS_PREFIX, QUANTITATIVE_ACTION_IDS_PREFIX],
                [action_specific_kwargs, quantitative_action_specific_kwargs],
            ):
                if keyword.startswith(prefix) and type(argument) is dict:
                    kwargs.pop(keyword)
                    inner_keyword = keyword.split(prefix)[1]
                    for action_id, value in argument.items():
                        target_kwargs[action_id][inner_keyword] = value
        return dict(action_specific_kwargs), dict(quantitative_action_specific_kwargs), kwargs

    @classmethod
    def _extract_action_model_class_and_attributes(
        cls, kwargs
    ) -> Tuple[Callable, Callable, Dict[str, Dict], Dict[str, Dict]]:
        """
        Utility function to extract kwargs that are specific for each action when constructing the action model.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            Additional parameters for the mab and for the action model.

        Returns
        -------
        action_model_cold_start : Callable
            Function handle for factoring the required action model.
        quantitative_action_model_cold_start : Callable
            Function handle for factoring the required quantitative action model.
        action_general_kwargs : Dict[str, any]
            Dictionary of parameters and their values for the action model.
        quantitative_action_general_kwargs : Dict[str, any]
            Dictionary of parameters and their values for the quantitative action model.
        """
        action_model_type = get_args(cls.model_fields["actions"].annotation)[1]
        action_model_classes = (
            get_args(action_model_type) if get_origin(action_model_type) is Union else (action_model_type,)
        )
        if len(action_model_classes) > 2:
            raise ValueError("Only up to two types of action models are supported.")
        quantitative_model_cold_start = model_cold_start = lambda **kwargs: None  # dummy callable
        action_general_kwargs = quantitative_action_general_kwargs = None
        for action_model_class in action_model_classes:
            if hasattr(action_model_class, "cold_start"):
                action_model_cold_start = action_model_class.cold_start
                action_model_attributes = extract_argument_names(action_model_cold_start)
                # cover for cold_start kwargs
                action_model_attributes = action_model_attributes + extract_argument_names(action_model_class)
            else:
                action_model_cold_start = action_model_class
                action_model_attributes = extract_argument_names(action_model_cold_start)
            general_kwargs = {k: kwargs.pop(k) for k in action_model_attributes if k in kwargs.keys()}

            if issubclass(action_model_class, (Model, ModelMO)):
                model_cold_start = action_model_cold_start
                action_general_kwargs = general_kwargs
            elif issubclass(action_model_class, QuantitativeModel):
                quantitative_model_cold_start = action_model_cold_start
                quantitative_action_general_kwargs = general_kwargs
            else:
                raise ValueError(f"Unsupported action model class: {action_model_class}")

        return (
            model_cold_start,
            quantitative_model_cold_start,
            action_general_kwargs,
            quantitative_action_general_kwargs,
        )
