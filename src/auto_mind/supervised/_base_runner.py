# ruff: noqa: E741 (ambiguous variable name)
import time
import math
import typing
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from auto_mind.supervised._action_data import Scheduler, BatchInOutParams, GeneralHookParams
from auto_mind.supervised._action_handlers import (
    BatchExecutor, BatchAccuracyCalculator, BatchExecutorParams, BatchAccuracyParams)
from auto_mind.supervised._batch_handler import (
    BatchHandlerData, MetricsHandler, MetricsHandlerInput, BatchHandler, BatchHandlerResult,
    BatchHandlerRunParams)

I = typing.TypeVar("I", bound=typing.Sized)
O = typing.TypeVar("O")
TG = typing.TypeVar("TG", bound=typing.Sized)
MT = typing.TypeVar("MT")

class BaseRunnerParams(typing.Generic[I, O, TG]):
    def __init__(
        self,
        epoch: int,
        is_train: bool,
        dataloader: DataLoader[tuple[I, TG]],
        model: torch.nn.Module,
        criterion: torch.nn.Module | typing.Callable[[BatchInOutParams[I, O, TG]], torch.Tensor],
        optimizer: optim.Optimizer | None,
        scheduler: Scheduler | None,
        clip_grad_max: float | None,
        hook: typing.Callable[[GeneralHookParams[I, O, TG]], None] | None,
        step_only_on_accuracy_loss: bool,
        batch_handler: BatchHandler,
    ):
        self.epoch = epoch
        self.is_train = is_train
        self.dataloader = dataloader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip_grad_max = clip_grad_max
        self.hook = hook
        self.step_only_on_accuracy_loss = step_only_on_accuracy_loss
        self.batch_handler = batch_handler

class BaseRunner(typing.Generic[I, O, TG, MT]):
    """
    Base class for running training and evaluation loops.

    This class provides the basic structure and methods for running training
    and evaluation loops in machine learning workflows. It handles the
    initialization, execution, and finalization of these loops, and can be
    extended to implement specific training and evaluation logic.

    Methods:
        run() -> BatchHandlerResult:
            Runs the training and evaluation loops.
    """
    def __init__(
        self,
        random_seed: int | None,
        executor: BatchExecutor[I, O],
        accuracy_calculator: BatchAccuracyCalculator[I, O, TG] | None,
        metrics_handler: MetricsHandler[I, O, TG, MT] | None,
    ):
        self._metrics_handler = metrics_handler
        self._random_seed = random_seed
        self._executor = executor
        self._accuracy_calculator = accuracy_calculator

    def run(self, params: BaseRunnerParams[I, O, TG]) -> BatchHandlerResult:
        epoch = params.epoch
        is_train = params.is_train
        dataloader = params.dataloader
        model = params.model
        criterion = params.criterion
        optimizer = params.optimizer
        scheduler = params.scheduler
        clip_grad_max = params.clip_grad_max
        hook = params.hook
        step_only_on_accuracy_loss = params.step_only_on_accuracy_loss
        batch_handler = params.batch_handler

        if is_train:
            model.train()

            if not optimizer:
                raise Exception('optimizer is not defined')
        else:
            model.eval()

        batch_handler.verify_early_stop()

        random_seed = (self._random_seed or 1) * (epoch + 1)
        torch.manual_seed(random_seed)

        def get_len(dataloader: typing.Iterable[I]) -> int | None:
            if isinstance(dataloader, typing.Sized):
                try:
                    return len(dataloader)
                except TypeError:
                    return None
            return None

        batch = 0
        total_batch = get_len(dataloader)
        current_amount = 0
        batch_metrics: typing.Any | None = None
        start_time = time.time()
        out: BatchHandlerData[I, typing.Any, typing.Any] | None = None

        metrics_handler = self._metrics_handler

        def update_batch_results(last: bool) -> None:
            nonlocal start_time
            nonlocal batch_metrics

            assert out is not None

            loss_value = out.loss
            batch_accuracy = out.accuracy

            end_time = time.time()
            time_diff = _time_diff_millis(start_time, end_time)
            start_time = end_time

            if metrics_handler:
                metrics_params = MetricsHandlerInput(
                    out=out,
                    time_diff=time_diff)
                batch_metrics = metrics_handler.define(metrics_params)

            with torch.no_grad():
                batch_handler.run(
                    batch=batch,
                    total_batch=total_batch,
                    amount=current_amount,
                    last=last,
                    loss=loss_value,
                    accuracy=batch_accuracy,
                    time_diff=time_diff,
                    batch_metrics=batch_metrics)

        for data in dataloader:
            if out is not None and batch > 0:
                update_batch_results(last=False)

            batch_handler.verify_early_stop()
            batch += 1

            if batch_handler.skip(batch):
                # skip the batch (was already processed previously)
                continue

            out = self._run_batch(
                params=BatchHandlerRunParams(
                    data=data,
                    batch=batch,
                    amount=batch_handler.amount),
                epoch=epoch,
                is_train=is_train,
                hook=hook,
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                clip_grad_max=clip_grad_max,
            )

            current_amount = out.amount

            if not current_amount:
                break

        amount = current_amount + batch_handler.amount

        if not amount:
            raise Exception('The dataloader is empty')

        if out is not None:
            update_batch_results(last=True)

        total_loss = batch_handler.total_loss / batch_handler.amount
        total_accuracy = (
            (batch_handler.total_accuracy / batch_handler.amount)
            if batch_handler.total_accuracy is not None
            else None)

        result = BatchHandlerResult(
            total_loss=total_loss,
            total_accuracy=total_accuracy,
            total_time=batch_handler.total_time,
            total_metrics=batch_handler.total_metrics)

        if scheduler:
            new_accuracy = result.total_accuracy

            if new_accuracy is None:
                scheduler.step()
            else:
                best_accuracy = batch_handler.best_accuracy
                worse_accuracy = (
                    new_accuracy is not None
                    and
                    best_accuracy is not None
                    and
                    best_accuracy > new_accuracy)

                if worse_accuracy or not step_only_on_accuracy_loss:
                    scheduler.step()

        return result

    def _run_batch(
        self,
        params: BatchHandlerRunParams[tuple[I, TG]],
        epoch: int,
        is_train: bool,
        model: torch.nn.Module,
        criterion: torch.nn.Module | typing.Callable[[BatchInOutParams[I, O, TG]], torch.Tensor],
        optimizer: optim.Optimizer | None,
        clip_grad_max: float | None,
        hook: typing.Callable[[GeneralHookParams[I, O, TG]], None] | None,
    ) -> BatchHandlerData[I, O, TG]:
        executor = self._executor
        accuracy_calculator = self._accuracy_calculator

        data = params.data
        batch = params.batch
        amount = params.amount

        input_batch, target_batch = data
        current_amount: int = len(target_batch)

        if not current_amount:
            raise Exception('Empty batch')

        if is_train and optimizer:
            optimizer.zero_grad()

        # call the model with the input and retrieve the output
        executor_params = BatchExecutorParams(
            model=model,
            input=input_batch)
        output = executor.run(executor_params)

        if isinstance(criterion, nn.Module):
            loss: torch.Tensor = criterion(output, target_batch)
        else:
            loss_params = BatchInOutParams(
                input=input_batch,
                output=output,
                target=target_batch)
            loss = criterion(loss_params)

        loss_value = loss.item()

        if math.isnan(loss_value):
            raise Exception('The loss is NaN')

        if is_train:
            loss.backward() # type: ignore

            if clip_grad_max is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    clip_grad_max)

            if optimizer:
                optimizer.step()

        with torch.no_grad():
            if accuracy_calculator is not None:
                accuracy_params = BatchAccuracyParams(
                    input=input_batch,
                    output=output,
                    target=target_batch)
                batch_accuracy = accuracy_calculator.run(accuracy_params)
            else:
                batch_accuracy = None

            if hook:
                hook(GeneralHookParams(
                    epoch=epoch,
                    batch=batch,
                    current_amount=amount + current_amount,
                    loss=loss_value,
                    accuracy=batch_accuracy,
                    target=target_batch,
                    input=input_batch,
                    output=output))

        return BatchHandlerData(
            amount=current_amount,
            loss=loss_value,
            accuracy=batch_accuracy,
            input=input_batch,
            output=output,
            target=target_batch)

####################################################
################ Private Functions #################
####################################################

def _time_diff_millis(start_time: float, end_time: float) -> int:
    return int((end_time - start_time) * 1000)
