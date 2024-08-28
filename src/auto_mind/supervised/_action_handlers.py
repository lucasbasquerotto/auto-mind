# ruff: noqa: E741 (ambiguous variable name)
import typing
import torch
import numpy as np
from torch import Tensor, optim, nn
from auto_mind.supervised._action_data import (
    GeneralEvalBaseResult, GeneralEvalResult, BatchInOutParams,
    StateWithMetrics, EarlyStopper, TrainEarlyStopper)

I = typing.TypeVar("I", bound=typing.Sized)
O = typing.TypeVar('O')
T = typing.TypeVar('T')
P = typing.TypeVar('P')
TG = typing.TypeVar("TG", bound=typing.Sized)
MT = typing.TypeVar('MT')
EI = typing.TypeVar("EI")
EO = typing.TypeVar("EO")

class AbortedException(Exception):
    pass

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

class BatchExecutorParams(typing.Generic[I]):
    def __init__(
        self,
        model: nn.Module,
        input: I,
    ):
        self.model = model
        self.input = input

class BatchExecutor(typing.Generic[I, O]):
    def run(self, params: BatchExecutorParams[I]) -> O:
        raise NotImplementedError

class GeneralBatchExecutor(BatchExecutor[Tensor, Tensor]):
    def run(self, params: BatchExecutorParams[Tensor]) -> Tensor:
        result: Tensor = params.model(params.input)
        return result

class BatchAccuracyParams(BatchInOutParams[I, O, TG], typing.Generic[I, O, TG]):
    pass

class BatchAccuracyCalculator(typing.Generic[I, O, TG]):
    def run(self, params: BatchAccuracyParams[I, O, TG]) -> float:
        raise NotImplementedError

class GeneralBatchAccuracyCalculator(
    BatchAccuracyCalculator[I, torch.Tensor, torch.Tensor],
    typing.Generic[I],
):
    def run(self, params: BatchAccuracyParams[I, torch.Tensor, torch.Tensor]) -> float:
        return (params.output.argmax(dim=1) == params.target).sum().item() / params.target.shape[0]

class MultiLabelBatchAccuracyCalculator(
    BatchAccuracyCalculator[I, torch.Tensor, torch.Tensor],
    typing.Generic[I],
):
    def run(self, params: BatchAccuracyParams[I, torch.Tensor, torch.Tensor]) -> float:
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

class ValueBatchAccuracyCalculator(
    BatchAccuracyCalculator[I, torch.Tensor, torch.Tensor],
    typing.Generic[I],
):
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

    def run(self, params: BatchAccuracyParams[I, torch.Tensor, torch.Tensor]) -> float:
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
            output=params.output,
            prediction=prediction,
            confidence=confidence)

class NoOutputEvaluator(LambdaOutputEvaluator[I, O, None], typing.Generic[I, O]):
    def __init__(self) -> None:
        super().__init__(lambda _: None)

class DefaultEvaluator(Evaluator[I, GeneralEvalResult[torch.Tensor, T]], typing.Generic[I, T]):
    def __init__(
        self,
        executor: BatchExecutor[I, torch.Tensor],
        output_evaluator: OutputEvaluator[I, torch.Tensor, T],
        random_mode: bool = False,
    ) -> None:
        self.executor = executor
        self.output_evaluator = output_evaluator
        self.random_mode = random_mode

    def run(self, params: EvaluatorParams[I]) -> GeneralEvalResult[torch.Tensor, T]:
        model = params.model
        input = params.input

        executor = self.executor
        output_evaluator = self.output_evaluator
        random_mode = self.random_mode

        if random_mode:
            model.train()
        else:
            model.eval()

        executor_params = BatchExecutorParams(
            model=model,
            input=input)
        output = executor.run(executor_params)

        default_result = GeneralEvalBaseResult(
            input=input,
            output=output)

        result = output_evaluator.run(default_result)

        return result

    def confidence(self, params: GeneralEvalBaseResult[I, torch.Tensor]) -> float:
        raise NotImplementedError

    @classmethod
    def single_result(cls, params: GeneralEvalBaseResult[I, torch.Tensor]) -> list[float]:
        out_data: list[float] = list(params.output.detach().numpy()[0])
        return out_data

class EvaluatorWithSimilarity(DefaultEvaluator[I, T], typing.Generic[I, T, P]):
    def similarity(self, predicted: P, expected: P) -> float:
        raise NotImplementedError

class MaxProbEvaluator(
    EvaluatorWithSimilarity[I, tuple[float, int], int],
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
    EvaluatorWithSimilarity[I, list[tuple[float, int]], int],
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
        out = params.output.detach().numpy()
        return [(out[i][argmax], argmax) for i, argmax in enumerate(np.argmax(out, axis=1))]

    def confidence(self, params: GeneralEvalBaseResult[I, torch.Tensor]) -> float:
        results = self.evaluate(params)
        value = np.mean([v for v, _ in results])
        return float(value)

    def similarity(self, predicted: int, expected: int) -> float:
        return 1.0 if predicted == expected else 0.0

class AllProbsEvaluator(
    EvaluatorWithSimilarity[I, list[float], list[float]],
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
    EvaluatorWithSimilarity[I, list[float], list[float]],
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
