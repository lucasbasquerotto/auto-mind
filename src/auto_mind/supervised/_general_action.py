# ruff: noqa: E741 (ambiguous variable name)
import math
import typing
import torch
import numpy as np
from torch import Tensor, optim, nn
from torch.utils.data import DataLoader
from auto_mind.supervised._action import (
    AbstractAction, MinimalHookParams, Scheduler, SingleModelMinimalEvalParams,
    SingleModelTestParams, SingleModelTrainParams, BatchInOutParams)
from auto_mind.supervised._action_impl import (
    ActionWrapper, SingleModelStateHandler, BatchHandler, BatchHandlerData, BatchHandlerRunParams,
    MinimalStateWithMetrics, MetricsHandler)

I = typing.TypeVar('I')
O = typing.TypeVar('O')
T = typing.TypeVar('T')
P = typing.TypeVar('P')
EI = typing.TypeVar("EI")
EO = typing.TypeVar("EO")

####################################################
################## General Action ##################
####################################################

class GeneralHookParams(MinimalHookParams, typing.Generic[I, O]):
    def __init__(
            self,
            epoch: int,
            batch: int,
            current_amount: int,
            loss: float,
            accuracy: float,
            target: Tensor,
            input: I,
            full_output: O | None,
            output: Tensor):

        super().__init__(
            current_amount=current_amount,
            loss=loss,
            accuracy=accuracy)

        self.epoch = epoch
        self.batch = batch
        self.target = target
        self.input = input
        self.output = output
        self.full_output = full_output

class GeneralEvalBaseResult(typing.Generic[I, O]):
    def __init__(
        self,
        input: I,
        full_output: O,
        main_output: Tensor,
    ):
        self.input = input
        self.full_output = full_output
        self.main_output = main_output

class GeneralEvalResult(typing.Generic[O, T]):
    def __init__(self, full_output: O, main_output: Tensor, prediction: T, confidence: float):
        self.full_output = full_output
        self.main_output = main_output
        self.prediction = prediction
        self.confidence = confidence

class GeneralTrainParams(
    SingleModelTrainParams[
        tuple[I, torch.Tensor],
        O,
        GeneralHookParams,
        GeneralHookParams,
    ],
    typing.Generic[I, O],
):
    pass

class GeneralTestParams(
    SingleModelTestParams[
        tuple[I, torch.Tensor],
        O,
        GeneralHookParams,
    ],
    typing.Generic[I, O],
):
    pass

class GeneralAction(
    AbstractAction[
        GeneralTrainParams[I, O],
        GeneralTestParams[I, O],
        MinimalStateWithMetrics,
    ],
    typing.Generic[I, O, T],
):
    pass

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
        return params.model(params.input)

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

class GeneralBatchAccuracyCalculator(BatchAccuracyCalculator):
    def run(self, params: BatchAccuracyParams) -> float:
        return (params.output.argmax(dim=1) == params.target).sum().item() / params.target.shape[0]

class MultiLabelBatchAccuracyCalculator(BatchAccuracyCalculator):
    def run(self, params: BatchAccuracyParams) -> float:
        # params.output shape: [batch, classes]
        # params.target shape: [batch, classes]
        # for each item compare how close they are, with value 1 if they are the same,
        # and 0 if the distance is 1 or more, or the value is outside the range [0, 1]
        differences = (params.output - params.target).abs()
        accuracies = 1.0 - torch.min(differences, torch.ones_like(differences))
        accuracies **= 2
        grouped = accuracies.sum(dim=0) / accuracies.shape[0]
        result = grouped.sum().item() / grouped.shape[0]
        return result

class ValueBatchAccuracyCalculator(BatchAccuracyCalculator):
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
    def __init__(self, error_margin=0.5, epsilon=1e-7):
        self.error_margin = error_margin
        self.epsilon = epsilon

    def run(self, params: BatchAccuracyParams) -> float:
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
        return accuracy.sum().item() / params.target.shape[0]

class MetricsCalculatorParams:
    def __init__(
        self,
        info: MinimalStateWithMetrics,
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
    def __init__(self):
        super().__init__(lambda _: None)

class DefaultEvaluator(Evaluator[I, GeneralEvalResult[O, T]], typing.Generic[I, O, T]):
    def __init__(
        self,
        executor: BatchExecutor[I, O],
        output_evaluator: OutputEvaluator[I, O, T] = NoOutputEvaluator(),
        random_mode=False,
    ):
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

        executor_params = BatchExecutorParams(
            model=model,
            input=input,
            last_output=None)
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
    def __init__(self, executor: BatchExecutor[I, torch.Tensor], random_mode=False):
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
    def __init__(self, executor: BatchExecutor[I, torch.Tensor], random_mode=False):
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
        return value

    def similarity(self, predicted: int, expected: int) -> float:
        return 1.0 if predicted == expected else 0.0

class AllProbsEvaluator(
    EvaluatorWithSimilarity[I, torch.Tensor, list[float], list[float]],
    typing.Generic[I],
):
    def __init__(self, executor: BatchExecutor[I, torch.Tensor], epsilon=1e-7, random_mode=False):
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
        return similarity.mean()

class ValuesEvaluator(
    EvaluatorWithSimilarity[I, torch.Tensor, list[float], list[float]],
    typing.Generic[I],
):
    def __init__(
        self,
        executor: BatchExecutor[I, torch.Tensor],
        log=False,
        error_margin=0.5,
        epsilon=1e-7,
        random_mode=False,
    ):
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
        return np.sum(accuracy) / len(accuracy)

####################################################
################### General Impl ###################
####################################################

class GeneralActionImpl(GeneralAction[I, O, T], typing.Generic[I, O, T]):
    def __init__(
        self,
        random_seed: int | None,
        use_best: bool,
        executor: BatchExecutor[I, O],
        accuracy_calculator: BatchAccuracyCalculator,
        metrics_handler: MetricsHandler | None,
        metrics_calculator: MetricsCalculator | None = None,
    ):
        main_state_handler = SingleModelStateHandler[
            GeneralTrainParams[I, O],
            GeneralTestParams[I, O],
        ](use_best=use_best)

        action_wrapper = ActionWrapper(
            state_handler=main_state_handler,
            train_epoch=lambda params: GeneralActionRunner(
                train=True,
                batch_handler=params.batch_handler,
                dataloader=params.main_params.train_dataloader,
                hook=params.main_params.train_hook,
                model=params.main_params.model,
                optimizer=params.main_params.optimizer,
                scheduler=(
                    None
                    if params.main_params.validation_dataloader
                    is not None else params.main_params.scheduler),
                criterion=params.main_params.criterion,
                step_only_on_accuracy_loss=params.main_params.step_only_on_accuracy_loss,
                clip_grad_max=params.main_params.clip_grad_max,
                epoch=params.epoch,
                random_seed=random_seed,
                executor=executor,
                accuracy_calculator=accuracy_calculator,
            ).run(),
            validate=lambda params: GeneralActionRunner(
                train=False,
                batch_handler=params.batch_handler,
                dataloader=params.main_params.validation_dataloader,
                hook=params.main_params.validation_hook,
                model=params.main_params.model,
                optimizer=None,
                scheduler=params.main_params.scheduler,
                criterion=params.main_params.criterion,
                step_only_on_accuracy_loss=params.main_params.step_only_on_accuracy_loss,
                clip_grad_max=None,
                epoch=params.epoch,
                random_seed=random_seed,
                executor=executor,
                accuracy_calculator=accuracy_calculator,
            ).run() if params.main_params.validation_dataloader is not None else None,
            test_inner=lambda params: GeneralActionRunner(
                train=False,
                batch_handler=params.batch_handler,
                dataloader=params.main_params.dataloader,
                hook=params.main_params.hook,
                model=params.main_params.model,
                optimizer=None,
                scheduler=None,
                criterion=params.main_params.criterion,
                step_only_on_accuracy_loss=False,
                clip_grad_max=None,
                epoch=params.epoch,
                random_seed=random_seed,
                executor=executor,
                accuracy_calculator=accuracy_calculator,
            ).run(),
            metrics_handler=metrics_handler,
        )

        def load_eval_state(params: SingleModelMinimalEvalParams) -> None:
            main_state_handler.load_eval_state(params)

        super().__init__(
            train=action_wrapper.train,
            test=action_wrapper.test,
            info=main_state_handler.info,
            load_eval_state=load_eval_state,
        )

        self.main_state_handler = main_state_handler
        self.metrics_calculator = metrics_calculator

    def calculate_metrics(
        self,
        params: MetricsCalculatorInputParams,
    ) -> dict[str, typing.Any] | None:
        main_state_handler = self.main_state_handler
        metrics_calculator = self.metrics_calculator
        save_path = params.save_path

        if not metrics_calculator or not save_path:
            return None

        info = main_state_handler.load_state_with_metrics(save_path=save_path)

        if not info:
            return None

        calc_params = MetricsCalculatorParams(info=info, model=params.model)
        metrics = metrics_calculator.run(calc_params)

        main_state_handler.save_metrics(metrics, save_path=save_path)

        return metrics

    def define_as_pending(self, save_path: str):
        self.main_state_handler.define_as_completed(completed=False, save_path=save_path)

    def define_as_completed(self, save_path: str):
        self.main_state_handler.define_as_completed(completed=True, save_path=save_path)

class GeneralActionRunner(typing.Generic[I, O, T]):
    def __init__(
        self,
        train: bool,
        dataloader: DataLoader[tuple[Tensor, Tensor]],
        hook: typing.Callable[[GeneralHookParams[I, O]], None] | None,
        model: nn.Module,
        optimizer: optim.Optimizer | None,
        scheduler: Scheduler | None,
        criterion: nn.Module | typing.Callable[[BatchInOutParams], torch.Tensor],
        step_only_on_accuracy_loss: bool,
        clip_grad_max: float | None,
        epoch: int,
        random_seed: int | None,
        executor: BatchExecutor[I, O],
        accuracy_calculator: BatchAccuracyCalculator,
        batch_handler: BatchHandler,
    ):
        last_output: O | None = None

        self.train = train
        self.dataloader = dataloader
        self.hook = hook
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.step_only_on_accuracy_loss = step_only_on_accuracy_loss
        self.clip_grad_max = clip_grad_max
        self.epoch = epoch
        self.random_seed = random_seed
        self.executor = executor
        self.accuracy_calculator = accuracy_calculator
        self.batch_handler = batch_handler
        self.best_accuracy = batch_handler.best_accuracy

        self.last_output = last_output

    def run_batch(self, params: BatchHandlerRunParams[tuple[I, torch.Tensor]]) -> BatchHandlerData:
        train = self.train
        hook = self.hook
        model = self.model
        optimizer = self.optimizer
        criterion = self.criterion
        clip_grad_max = self.clip_grad_max
        epoch = self.epoch
        executor = self.executor
        accuracy_calculator = self.accuracy_calculator

        data = params.data
        batch = params.batch
        amount = params.amount

        input_batch, target_batch = data
        current_amount: int = len(target_batch)

        if not current_amount:
            raise Exception('Empty batch')

        if train and optimizer:
            optimizer.zero_grad()

        # call the model with the input and retrieve the output
        executor_params = BatchExecutorParams(
            model=model,
            input=input_batch,
            last_output=self.last_output)
        full_output = executor.run(executor_params)
        output = executor.main_output(full_output)
        self.last_output = full_output

        if isinstance(criterion, nn.Module):
            loss: Tensor = criterion(output, target_batch)
        else:
            params = BatchAccuracyParams(
                input=input_batch,
                full_output=full_output,
                output=output,
                target=target_batch)
            loss = criterion(params)

        loss_value = loss.item()

        if math.isnan(loss_value):
            raise Exception('The loss is NaN')

        if train :
            loss.backward()

            if clip_grad_max is not None:
                torch.nn.utils.clip_grad_norm_( # type: ignore
                    model.parameters(),
                    clip_grad_max)

            if optimizer:
                optimizer.step()

        with torch.no_grad():
            accuracy_params = BatchAccuracyParams(
                input=input_batch,
                full_output=full_output,
                output=output,
                target=target_batch)
            batch_accuracy = accuracy_calculator.run(accuracy_params)

            if hook:
                hook(GeneralHookParams(
                    epoch=epoch,
                    batch=batch,
                    current_amount=amount + current_amount,
                    loss=loss_value,
                    accuracy=batch_accuracy,
                    target=target_batch,
                    input=input_batch,
                    output=output,
                    full_output=self.last_output))

        return BatchHandlerData(
            amount=current_amount,
            loss=loss_value,
            accuracy=batch_accuracy,
            input=input_batch,
            full_output=full_output,
            output=output,
            target=target_batch)

    def run(self):
        train = self.train
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        step_only_on_accuracy_loss = self.step_only_on_accuracy_loss

        if train:
            model.train()

            if not optimizer:
                raise Exception('optimizer is not defined')
        else:
            model.eval()

        self.last_output = None

        result = self.batch_handler.run(
            dataloader=self.dataloader,
            fn=self.run_batch,
            epoch=self.epoch,
            random_seed=self.random_seed)

        if scheduler:
            new_accuracy = result.total_accuracy
            best_accuracy = self.best_accuracy
            worse_accuracy = best_accuracy is not None and best_accuracy > new_accuracy

            if worse_accuracy or not step_only_on_accuracy_loss:
                scheduler.step()

            if best_accuracy is None or not worse_accuracy:
                self.best_accuracy = new_accuracy

        return result
