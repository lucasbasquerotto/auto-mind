import os
import typing
import warnings
import torch
from auto_mind.supervised._action_data import (
    EvalParams, TrainResult, TestResult,StateWithMetrics,
    TestParams, TrainParams, BaseResult, EvalState, FullState,
    MinimalEvalParams)

S = typing.TypeVar("S", bound=BaseResult)
PE = typing.TypeVar("PE", bound=MinimalEvalParams)
ATR = typing.TypeVar("ATR", bound=TrainParams[
    typing.Any, typing.Any, typing.Any, typing.Any])
ATE = typing.TypeVar("ATE", bound=TestParams[
    typing.Any, typing.Any, typing.Any, typing.Any])

class StateHandler(typing.Generic[ATR, ATE]):
    def __init__(self, use_best: bool):
        def get_eval_state(
            params: EvalParams,
            state_dict: dict[str, typing.Any],
        ) -> EvalState:
            return EvalState.from_state_dict_with_params(
                params,
                use_best=use_best,
                state_dict=state_dict)

        def test_state_from_dict(
            params: ATE,
            state_dict: dict[str, typing.Any],
        ) -> EvalState | None:
            return get_eval_state(
                params=EvalParams(
                    model=params.model,
                    save_path=params.save_path,
                    skip_load_state=params.skip_load_state,
                ),
                state_dict=state_dict,
            )

        def info_from_dict(state_dict: dict[str, typing.Any]) -> StateWithMetrics:
            return StateWithMetrics.from_state_dict(state_dict)

        def train_state_from_dict(
            params: ATR,
            state_dict: dict[str, typing.Any],
        ) -> FullState:
            return FullState.from_state_dict_with_params(
                params,
                use_best=False,
                state_dict=state_dict)

        def new_train_state(
            params: ATR,
            train_results: TrainResult,
            last_state_dict: dict[str, typing.Any] | None,
        ) -> FullState:
            return FullState(
                model=params.model,
                optimizer=params.optimizer,
                scheduler=params.scheduler,
                early_stopper=params.early_stopper,
                train_results=train_results,
                best_state_dict=None,
                test_results=(
                    TestResult.from_state_dict(last_state_dict['test_results'])
                    if last_state_dict and last_state_dict.get('test_results')
                    else None),
                metrics=last_state_dict.get('metrics') if last_state_dict else None,
            )

        def new_test_state(
            params: ATE,
            test_results: TestResult,
            last_state_dict: dict[str, typing.Any] | None,
        ) -> EvalState | None:
            return EvalState(
                model=params.model,
                train_results=TrainResult.from_state_dict(
                    last_state_dict['train_results']),
                test_results=test_results,
                metrics=last_state_dict.get('metrics'),
            ) if last_state_dict and last_state_dict.get('train_results') else None

        self._info_from_dict = info_from_dict
        self._train_state_from_dict = train_state_from_dict
        self._new_train_state = new_train_state
        self._test_state_from_dict = test_state_from_dict
        self._new_test_state = new_test_state
        self._get_eval_state = get_eval_state

    def _load_state(
        self,
        params: PE,
        get_state: typing.Callable[[PE, dict[str, typing.Any]], S | None] | None,
    ) -> tuple[S | None, dict[str, typing.Any] | None]:
        state: S | None = None
        state_dict: dict[str, typing.Any] | None = None

        if get_state and not params.skip_load_state:
            if not params.save_path:
                raise Exception(
                    'save_path is not defined, but skip_load_state is False')

            state_dict = _load_state_dict(save_path=params.save_path)
            state = get_state(params, state_dict) if state_dict else None

        return state, state_dict

    def _save_state(
        self,
        params: MinimalEvalParams,
        state: BaseResult | None,
        last_state_dict: dict[str, typing.Any] | None,
    ) -> None:
        if state and params.save_path:
            state_dict = state.state_dict()

            if last_state_dict:
                state_dict = last_state_dict | state_dict

            _save_state_dict(
                state_dict=state_dict,
                save_path=params.save_path)

    def _save_state_dict(self, save_path: str | None, state_dict: dict[str, typing.Any]) -> None:
        if state_dict and save_path:
            _save_state_dict(
                state_dict=state_dict,
                save_path=save_path)

    def info(self, save_path: str) -> StateWithMetrics | None:
        state_dict = _load_state_dict(save_path=save_path)
        info = self._info_from_dict(state_dict) if state_dict else None
        return info

    def load_train_state(
        self,
        params: ATR,
    ) -> tuple[FullState | None, dict[str, typing.Any] | None]:
        return self._load_state(params, self._train_state_from_dict)

    def save_train_state(
        self,
        params: ATR,
        result: TrainResult,
        last_state_dict: dict[str, typing.Any] | None,
    ) -> None:
        new_train_state = self._new_train_state

        if params.save_path:
            state = new_train_state(params, result, last_state_dict)
            self._save_state(
                params=params,
                state=state,
                last_state_dict=last_state_dict)

    def load_test_state(
        self,
        params: ATE,
    ) -> tuple[EvalState | None, dict[str, typing.Any] | None]:
        return self._load_state(params, self._test_state_from_dict)

    def save_test_state(
        self,
        params: ATE,
        result: TestResult,
        last_state_dict: dict[str, typing.Any] | None,
    ) -> None:
        new_test_state = self._new_test_state

        if params.save_path:
            state = new_test_state(params, result, last_state_dict)
            self._save_state(
                params=params,
                state=state,
                last_state_dict=last_state_dict)

    def load_eval_state(
        self,
        params: EvalParams,
    ) -> tuple[EvalState | None, dict[str, typing.Any] | None]:
        return self._load_state(params, self._get_eval_state)

    def load_state_with_metrics(self, save_path: str) -> StateWithMetrics | None:
        state_dict = _load_state_dict(save_path=save_path)
        return StateWithMetrics.from_state_dict(state_dict) if state_dict else None

    def save_metrics(self, metrics: dict[str, typing.Any], save_path: str | None) -> None:
        if save_path:
            last_state_dict = _load_state_dict(save_path=save_path)

            if last_state_dict:
                last_state_dict['metrics'] = metrics
                self._save_state_dict(save_path, last_state_dict)

    def define_as_completed(self, completed: bool, save_path: str | None) -> None:
        if save_path:
            last_state_dict = _load_state_dict(save_path=save_path)

            if last_state_dict:
                last_state_dict['completed'] = completed
                self._save_state_dict(save_path, last_state_dict)

####################################################
################ Private Functions #################
####################################################

def _load_state_dict(save_path: str | None) -> dict[str, typing.Any] | None:
    if save_path:
        if os.path.isfile(save_path):
            checkpoint: dict[str, typing.Any] | None = torch.load(save_path, weights_only=True)
            return checkpoint
    else:
        warnings.warn('load_state_dict skipped: save_path is not defined', UserWarning)

    return None

def _save_state_dict(
    state_dict: dict[str, typing.Any],
    save_path: str | None,
) -> dict[str, typing.Any] | None:
    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        torch.save(state_dict, save_path)

        return state_dict
    else:
        warnings.warn('save_state_dict skipped: save_path is not defined', UserWarning)

    return None
