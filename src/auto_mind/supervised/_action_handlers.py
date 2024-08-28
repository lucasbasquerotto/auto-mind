# ruff: noqa: E741 (ambiguous variable name)
import os
import typing
import warnings
import torch
import numpy as np
from torch import Tensor, optim, nn
from auto_mind.supervised._action_data import (
    GeneralEvalBaseResult, GeneralEvalResult, EvalParams,
    TestParams, TrainParams, BatchInOutParams, BaseResult,
    MinimalEvalParams, EvalState, FullState, TestResult,
    TrainResult, StateWithMetrics, EarlyStopper, TrainEarlyStopper)

I = typing.TypeVar('I')
O = typing.TypeVar('O')
T = typing.TypeVar('T')
P = typing.TypeVar('P')
TG = typing.TypeVar('TG')
MT = typing.TypeVar('MT')
EI = typing.TypeVar("EI")
EO = typing.TypeVar("EO")
S = typing.TypeVar("S", bound=BaseResult)

####################################################
############# Default Implementations ##############
####################################################

class ChainedEarlyStopper(TrainEarlyStopper[MT], typing.Generic[MT]):
    def __init__(self, stoppers: list[EarlyStopper]):
        self.stoppers = stoppers

    def check(self) -> bool:
        if self.check_finish():
            return True
        return any(stopper.check() for stopper in self.stoppers)

    def check_finish(self) -> bool:
        return (
            any(stopper.check_finish()
            for stopper in self.stoppers
            if isinstance(stopper, TrainEarlyStopper)))

    def update_epoch(self, loss: float, accuracy: float | None, metrics: MT | None) -> None:
        for stopper in self.stoppers:
            if isinstance(stopper, TrainEarlyStopper):
                stopper.update_epoch(loss=loss, accuracy=accuracy, metrics=metrics)

    def state_dict(self) -> dict[str, typing.Any]:
        return dict(stoppers=[stopper.state_dict() for stopper in self.stoppers])

    def load_state_dict(self, state_dict: dict[str, typing.Any]) -> typing.Self:
        for i, stopper in enumerate(self.stoppers):
            stopper.load_state_dict(state_dict['stoppers'][i])
        return self

class AccuracyEarlyStopper(TrainEarlyStopper[MT], typing.Generic[MT]):
    def __init__(self, min_accuracy: float, patience: int = 5):
        self.patience = patience
        self.min_accuracy = min_accuracy
        self.amount = 0

    def check_finish(self) -> bool:
        return self.amount >= self.patience

    def update_epoch(self, loss: float, accuracy: float | None, metrics: MT | None) -> None:
        if accuracy is None:
            self.amount = 0
        elif accuracy < self.min_accuracy:
            self.amount = 0
        else:
            self.amount += 1

    def state_dict(self) -> dict[str, typing.Any]:
        parent = super().state_dict()
        return dict(amount=self.amount, parent=parent)

    def load_state_dict(self, state_dict: dict[str, typing.Any]) -> typing.Self:
        self.amount = state_dict.get('amount', 0)
        super().load_state_dict(state_dict.get('parent', {}))
        return self

class OptimizerChain(optim.Optimizer):
    def __init__(self, optimizers: list[optim.Optimizer]) -> None:
        super().__init__(params=[], defaults=dict())
        self.optimizers: list[optim.Optimizer] = optimizers

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None) -> None: # type: ignore
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self) -> dict[str, typing.Any]:
        return {
            f'optimizer_{i}': optimizer.state_dict()
            for i, optimizer in enumerate(self.optimizers)
        }

    def load_state_dict(self, state_dict: dict[str, typing.Any]) -> None:
        for i, optimizer in enumerate(self.optimizers):
            optimizer.load_state_dict(state_dict[f'optimizer_{i}'])

####################################################
####### Executors, Calculators & Evaluators ########
####################################################

class BatchExecutorParams(typing.Generic[I, O]):
    def __init__(
        self,
        model: nn.Module,
        input: I,
        last_output: O | None,
    ):
        self.model = model
        self.input = input
        self.last_output = last_output

class BatchExecutor(typing.Generic[I, O]):
    def run(self, params: BatchExecutorParams[I, O]) -> O:
        raise NotImplementedError

    def main_output(self, output: O) -> Tensor:
        raise NotImplementedError

class GeneralBatchExecutor(BatchExecutor[Tensor, Tensor]):
    def run(self, params: BatchExecutorParams[Tensor, Tensor]) -> Tensor:
        result: Tensor = params.model(params.input)
        return result

    def main_output(self, output: Tensor) -> Tensor:
        return output

class BatchAccuracyParams(BatchInOutParams[I, O], typing.Generic[I, O]):
    def __init__(
        self,
        input: I,
        full_output: O,
        output: Tensor,
        target: Tensor,
    ):
        super().__init__(
            input=input,
            full_output=full_output,
            output=output,
            target=target)

class BatchAccuracyCalculator(typing.Generic[I, O]):
    def run(self, params: BatchAccuracyParams[I, O]) -> float:
        raise NotImplementedError

class GeneralBatchAccuracyCalculator(BatchAccuracyCalculator[I, O], typing.Generic[I, O]):
    def run(self, params: BatchAccuracyParams[I, O]) -> float:
        return (params.output.argmax(dim=1) == params.target).sum().item() / params.target.shape[0]

class MultiLabelBatchAccuracyCalculator(BatchAccuracyCalculator[I, O], typing.Generic[I, O]):
    def run(self, params: BatchAccuracyParams[I, O]) -> float:
        # params.output shape: [batch, classes]
        # params.target shape: [batch, classes]
        # for each item compare how close they are, with value 1 if they are the same,
        # and 0 if the distance is 1 or more, or the value is outside the range [0, 1]
        differences = (params.output - params.target).abs()
        accuracies = 1.0 - torch.min(differences, torch.ones_like(differences))
        accuracies **= 2
        grouped = accuracies.sum(dim=0) / accuracies.shape[0]
        result: float = grouped.sum().item() / grouped.shape[0]
        return result

class ValueBatchAccuracyCalculator(BatchAccuracyCalculator[I, O], typing.Generic[I, O]):
    """
    Calculates the accuracy of the output values in relation to the targets for continuous values.

    The accuracy is calculated as the percentage of values that are within a certain margin of error
    in relation to the targets.

    For example, if the error margin is 0.5, the accuracy will be calculated with max accuracy
    when the predicted value is the same as the target, and will decrease linearly for predicted
    values that are in a range within 50% of the target value. A target value of 100.0 will have
    an accuracy of 1.0 if the predicted value is 100.0, 0.5 if it's 75.0 or 125.0, and 0.0 if
    it's less than 50.0 or more than 150.0.

    Parameters
    ----------
    error_margin  : float
        The margin of error for the values in relation to the targets
    epsilon       : float
        A small value to avoid division by zero
    """
    def __init__(self, error_margin: float = 0.5, epsilon: float = 1e-7) -> None:
        self.error_margin = error_margin
        self.epsilon = epsilon

    def run(self, params: BatchAccuracyParams[I, O]) -> float:
        range_tensor = self.error_margin*params.target.abs() + self.epsilon

        # calculate the absolute difference between output and target
        difference = (params.output - params.target).abs()
        # The loss is the difference divided by the range, which gives 0.0
        # if the predicted value is the same as the target, 1.0 if it's
        # in the range limit of the error margin, and higher if it's outside
        loss = difference / range_tensor
        # cap the loss to 1.0
        loss = torch.min(loss, torch.ones_like(loss))
        # calculate the accuracy
        accuracy = torch.ones_like(loss) - loss
        # sum the correct values and divide by the batch size
        return float(accuracy.sum().item() / params.target.shape[0])

class MetricsCalculatorParams:
    def __init__(
        self,
        info: StateWithMetrics,
        model: torch.nn.Module,
    ):
        self.info = info
        self.model = model

class MetricsCalculatorInputParams:
    def __init__(
        self,
        model: torch.nn.Module,
        save_path: str | None,
    ):
        self.model = model
        self.save_path = save_path

class MetricsCalculator:
    def run(self, params: MetricsCalculatorParams) -> dict[str, typing.Any]:
        raise NotImplementedError

class EvaluatorParams(typing.Generic[EI]):
    def __init__(
        self,
        model: nn.Module,
        input: EI,
    ):
        self.model = model
        self.input = input

class Evaluator(typing.Generic[EI, EO]):
    def run(self, params: EvaluatorParams[EI]) -> EO:
        raise NotImplementedError()

class OutputEvaluator(typing.Generic[I, O, T]):
    def run(self, params: GeneralEvalBaseResult[I, O]) -> GeneralEvalResult[O, T]:
        raise NotImplementedError()

class LambdaOutputEvaluator(OutputEvaluator[I, O, T], typing.Generic[I, O, T]):
    def __init__(
        self,
        fn: typing.Callable[[GeneralEvalBaseResult[I, O]], T],
        fn_confidence: typing.Callable[[GeneralEvalBaseResult[I, O]], float] | None = None,
    ):
        self.fn = fn
        self.fn_confidence = fn_confidence

    def run(self, params: GeneralEvalBaseResult[I, O]) -> GeneralEvalResult[O, T]:
        prediction = self.fn(params)
        confidence = self.fn_confidence(params) if self.fn_confidence else 0.0
        return GeneralEvalResult(
            full_output=params.full_output,
            main_output=params.main_output,
            prediction=prediction,
            confidence=confidence)

class NoOutputEvaluator(LambdaOutputEvaluator[I, O, None], typing.Generic[I, O]):
    def __init__(self) -> None:
        super().__init__(lambda _: None)

class DefaultEvaluator(Evaluator[I, GeneralEvalResult[O, T]], typing.Generic[I, O, T]):
    def __init__(
        self,
        executor: BatchExecutor[I, O],
        output_evaluator: OutputEvaluator[I, O, T],
        random_mode: bool = False,
    ) -> None:
        self.executor = executor
        self.output_evaluator = output_evaluator
        self.random_mode = random_mode

    def run(self, params: EvaluatorParams[I]) -> GeneralEvalResult[O, T]:
        model = params.model
        input = params.input

        executor = self.executor
        output_evaluator = self.output_evaluator
        random_mode = self.random_mode

        if random_mode:
            model.train()
        else:
            model.eval()

        last_output: O | None = None
        executor_params = BatchExecutorParams(
            model=model,
            input=input,
            last_output=last_output)
        full_output = executor.run(executor_params)
        output = executor.main_output(full_output)

        default_result = GeneralEvalBaseResult(
            input=input,
            full_output=full_output,
            main_output=output)

        result = output_evaluator.run(default_result)

        return result

    def confidence(self, params: GeneralEvalBaseResult[I, O]) -> float:
        raise NotImplementedError

    @classmethod
    def single_result(cls, params: GeneralEvalBaseResult[I, O]) -> list[float]:
        out_data: list[float] = list(params.main_output.detach().numpy()[0])
        return out_data

class EvaluatorWithSimilarity(DefaultEvaluator[I, O, T], typing.Generic[I, O, T, P]):
    def similarity(self, predicted: P, expected: P) -> float:
        raise NotImplementedError

class MaxProbEvaluator(
    EvaluatorWithSimilarity[I, torch.Tensor, tuple[float, int], int],
    typing.Generic[I],
):
    def __init__(
        self,
        executor: BatchExecutor[I, torch.Tensor],
        random_mode: bool = False,
    ):
        super().__init__(
            executor=executor,
            output_evaluator=LambdaOutputEvaluator(
                fn=self.evaluate,
                fn_confidence=self.confidence),
            random_mode=random_mode)

    def evaluate(self, params: GeneralEvalBaseResult[I, torch.Tensor]) -> tuple[float, int]:
        out = self.single_result(params)
        argmax = int(np.argmax(out))
        value = out[argmax]
        return value, argmax

    def confidence(self, params: GeneralEvalBaseResult[I, torch.Tensor]) -> float:
        value, _ = self.evaluate(params)
        return value

    def similarity(self, predicted: int, expected: int) -> float:
        return 1.0 if predicted == expected else 0.0

class MaxProbBatchEvaluator(
    EvaluatorWithSimilarity[I, torch.Tensor, list[tuple[float, int]], int],
    typing.Generic[I],
):
    def __init__(
        self,
        executor: BatchExecutor[I, torch.Tensor],
        random_mode: bool = False,
    ) -> None:
        super().__init__(
            executor=executor,
            output_evaluator=LambdaOutputEvaluator(
                fn=self.evaluate,
                fn_confidence=self.confidence),
            random_mode=random_mode)

    def evaluate(self, params: GeneralEvalBaseResult[I, torch.Tensor]) -> list[tuple[float, int]]:
        out = params.main_output.detach().numpy()
        return [(out[i][argmax], argmax) for i, argmax in enumerate(np.argmax(out, axis=1))]

    def confidence(self, params: GeneralEvalBaseResult[I, torch.Tensor]) -> float:
        results = self.evaluate(params)
        value = np.mean([v for v, _ in results])
        return float(value)

    def similarity(self, predicted: int, expected: int) -> float:
        return 1.0 if predicted == expected else 0.0

class AllProbsEvaluator(
    EvaluatorWithSimilarity[I, torch.Tensor, list[float], list[float]],
    typing.Generic[I],
):
    def __init__(
        self,
        executor: BatchExecutor[I, torch.Tensor],
        epsilon: float = 1e-7,
        random_mode: bool = False,
    ) -> None:
        super().__init__(
            executor=executor,
            output_evaluator=LambdaOutputEvaluator(
                fn=self.evaluate,
                fn_confidence=self.confidence),
            random_mode=random_mode)
        self.epsilon = epsilon

    def evaluate(self, params: GeneralEvalBaseResult[I, torch.Tensor]) -> list[float]:
        result = self.single_result(params)
        return result

    def confidence(self, params: GeneralEvalBaseResult[I, torch.Tensor]) -> float:
        probs = self.evaluate(params)
        # for each probability, the confidence is how close it is
        # to 0 or 1, with no confidence (0.0) at 0.5, and
        # max confidence (1.0) at 0.0 or 1.0
        confidences = np.absolute(np.array(probs) - 0.5) * 2.0
        # the final confidence is the the smallest confidence
        confidence = float(confidences.min())
        return confidence

    def similarity(self, predicted: list[float], expected: list[float]) -> float:
        # calculate the absolute difference between the input and output
        difference = np.abs(np.array(predicted) - np.array(expected))
        # the similarity is the inverse of the difference
        similarity = 1.0 - difference
        return float(similarity.mean())

class ValuesEvaluator(
    EvaluatorWithSimilarity[I, torch.Tensor, list[float], list[float]],
    typing.Generic[I],
):
    def __init__(
        self,
        executor: BatchExecutor[I, torch.Tensor],
        log: bool = False,
        error_margin: float = 0.5,
        epsilon: float = 1e-7,
        random_mode: bool = False,
    ) -> None:
        super().__init__(
            executor=executor,
            output_evaluator=LambdaOutputEvaluator(
                fn=self.evaluate,
                fn_confidence=self.confidence),
            random_mode=random_mode)
        self.log = log
        self.error_margin = error_margin
        self.epsilon = epsilon

    def evaluate(self, params: GeneralEvalBaseResult[I, torch.Tensor]) -> list[float]:
        out = self.single_result(params)
        result = [
            value if not self.log else float(np.exp(value))
            for value in out
        ]
        return result

    def confidence(self, params: GeneralEvalBaseResult[I, torch.Tensor]) -> float:
        raise NotImplementedError

    def similarity(self, predicted: list[float], expected: list[float]) -> float:
        # the similarity must be based in ValueBatchAccuracyCalculator
        # the values may go from minus infinite to infinite, so the similarity
        # must be calculated based on the difference between the values
        # and the maximum error margin
        range_tensor = self.error_margin*np.abs(expected) + self.epsilon

        # calculate the absolute difference between output and target
        difference = np.abs(np.array(expected) - np.array(predicted))
        # The loss is the difference divided by the range, which gives 0.0
        # if the predicted value is the same as the target, 1.0 if it's
        # in the range limit of the error margin, and higher if it's outside
        loss = difference / range_tensor
        # cap the loss to 1.0
        loss = np.min(loss, np.array([1 for _ in loss]))
        # calculate the accuracy
        accuracy = np.array([1 for _ in loss]) - loss
        # sum the correct values and divide by the batch size
        return float(np.sum(accuracy) / len(accuracy))

####################################################
################## State Handler ###################
####################################################

PE = typing.TypeVar("PE", bound=MinimalEvalParams)
ATR = typing.TypeVar("ATR", bound=TrainParams[
    typing.Any, typing.Any, typing.Any])
ATE = typing.TypeVar("ATE", bound=TestParams[
    typing.Any, typing.Any, typing.Any])

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
