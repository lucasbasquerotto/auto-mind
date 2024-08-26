# ruff: noqa: E741 (ambiguous variable name)
from typing import Any, Dict, Generic, Protocol, TypeVar
import typing
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

####################################################
############### Protocols and Types ################
####################################################

M = typing.TypeVar("M")

class Scheduler(Protocol):
    def step(self, epoch: int | None = None) -> None:
        ...

    def state_dict(self) -> Dict[str, Any]:
        ...

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        ...

class EarlyStopper:
    def check(self) -> bool:
        return False

    def state_dict(self) -> dict[str, typing.Any]:
        return dict()

    def load_state_dict(
        self,
        state_dict: dict[str, typing.Any], # pylint: disable=unused-argument
    ) -> typing.Self:
        return self

class TrainEarlyStopper(EarlyStopper, typing.Generic[M]):
    def check(self) -> bool:
        return self.check_finish()

    def check_finish(self) -> bool:
        return False

    def update_epoch(self, loss: float, accuracy: float, metrics: M | None):
        pass

class ChainedEarlyStopper(TrainEarlyStopper[M], typing.Generic[M]):
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

    def update_epoch(self, loss: float, accuracy: float, metrics: M | None):
        for stopper in self.stoppers:
            if isinstance(stopper, TrainEarlyStopper):
                stopper.update_epoch(loss=loss, accuracy=accuracy, metrics=metrics)

    def state_dict(self) -> dict[str, typing.Any]:
        return dict(stoppers=[stopper.state_dict() for stopper in self.stoppers])

    def load_state_dict(self, state_dict: dict[str, typing.Any]) -> typing.Self:
        for i, stopper in enumerate(self.stoppers):
            stopper.load_state_dict(state_dict['stoppers'][i])
        return self

class AccuracyEarlyStopper(TrainEarlyStopper[M], typing.Generic[M]):
    def __init__(self, min_accuracy: float, patience: int = 5):
        self.patience = patience
        self.min_accuracy = min_accuracy
        self.amount = 0

    def check_finish(self) -> bool:
        return self.amount >= self.patience

    def update_epoch(self, loss: float, accuracy: float, metrics: M | None):
        if accuracy < self.min_accuracy:
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

    def step(self, closure=None) -> None:
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
################# General Results ##################
####################################################

class BaseResult():
    def state_dict(self) -> dict[str, typing.Any]: ...

class ExecutionCursor(BaseResult):
    def __init__(
        self,
        amount: int,
        total_loss: float,
        total_accuracy: float,
        total_time: int,
        total_metrics: typing.Any | None,
    ):
        self.amount = amount
        self.total_loss = total_loss
        self.total_accuracy = total_accuracy
        self.total_time = total_time
        self.total_metrics = total_metrics

    def state_dict(self) -> dict[str, typing.Any]:
        return dict(
            amount=self.amount,
            total_loss=self.total_loss,
            total_accuracy=self.total_accuracy,
            total_time=self.total_time,
            total_metrics=self.total_metrics,
        )

    @staticmethod
    def from_state_dict(state_props: dict[str, typing.Any]):
        return ExecutionCursor(
            amount=state_props['amount'],
            total_loss=state_props['total_loss'],
            total_accuracy=state_props['total_accuracy'],
            total_time=state_props['total_time'],
            total_metrics=state_props['total_metrics'],
        )

class TrainResult(BaseResult):
    def __init__(
        self,
        epoch: int,
        early_stopped: bool,
        early_stopped_max_epochs: int,
        train_batch: int,
        train_total_batch: int | None,
        val_batch: int,
        val_total_batch: int | None,
        last_loss: float,
        last_accuracy: float,
        last_metrics: typing.Any | None,
        last_train_loss: float,
        last_train_accuracy: float,
        last_train_metrics: typing.Any | None,
        last_val_loss: float,
        last_val_accuracy: float,
        last_val_metrics: typing.Any | None,
        best_epoch: int,
        best_accuracy: float,
        best_train_accuracy: float,
        best_val_accuracy: float,
        total_train_time: int,
        total_val_time: int,
        accuracies: list[tuple[int, float]],
        losses: list[tuple[int, float]],
        times: list[tuple[int, float]],
        metrics: list[tuple[int, typing.Any]] | None = None,
        val_accuracies: list[tuple[int, float]] | None = None,
        val_losses: list[tuple[int, float]] | None = None,
        val_times: list[tuple[int, float]] | None = None,
        val_metrics: list[tuple[int, typing.Any]] | None = None,
        last_epoch_accuracies: list[tuple[int, float]] | None = None,
        last_epoch_losses: list[tuple[int, float]] | None = None,
        last_epoch_times: list[tuple[int, int]] | None = None,
        last_epoch_metrics: list[tuple[int, typing.Any]] | None = None,
        last_epoch_val_accuracies: list[tuple[int, float]] | None = None,
        last_epoch_val_losses: list[tuple[int, float]] | None = None,
        last_epoch_val_times: list[tuple[int, int]] | None = None,
        last_epoch_val_metrics: list[tuple[int, typing.Any]] | None = None,
        epoch_cursor: ExecutionCursor | None = None,
        batch_train_cursor: ExecutionCursor | None = None,
        batch_val_cursor: ExecutionCursor | None = None,
    ):
        self.epoch = epoch
        self.early_stopped = early_stopped
        self.early_stopped_max_epochs = early_stopped_max_epochs
        self.train_batch = train_batch
        self.train_total_batch = train_total_batch
        self.val_batch = val_batch
        self.val_total_batch = val_total_batch
        self.last_train_loss = last_train_loss
        self.last_train_accuracy = last_train_accuracy
        self.last_train_metrics = last_train_metrics
        self.last_val_loss = last_val_loss
        self.last_val_accuracy = last_val_accuracy
        self.last_val_metrics = last_val_metrics
        self.last_loss = last_loss
        self.last_accuracy = last_accuracy
        self.last_metrics = last_metrics
        self.best_epoch = best_epoch
        self.best_accuracy = best_accuracy
        self.best_train_accuracy = best_train_accuracy
        self.best_val_accuracy = best_val_accuracy
        self.total_train_time = total_train_time
        self.total_val_time = total_val_time
        self.accuracies = accuracies
        self.losses = losses
        self.times = times
        self.metrics = metrics
        self.val_accuracies = val_accuracies
        self.val_losses = val_losses
        self.val_times = val_times
        self.val_metrics = val_metrics
        self.last_epoch_accuracies = last_epoch_accuracies
        self.last_epoch_losses = last_epoch_losses
        self.last_epoch_times = last_epoch_times
        self.last_epoch_metrics = last_epoch_metrics
        self.last_epoch_val_accuracies = last_epoch_val_accuracies
        self.last_epoch_val_losses = last_epoch_val_losses
        self.last_epoch_val_times = last_epoch_val_times
        self.last_epoch_val_metrics = last_epoch_val_metrics
        self.epoch_cursor = epoch_cursor
        self.batch_train_cursor = batch_train_cursor
        self.batch_val_cursor = batch_val_cursor

    def state_dict(self) -> dict[str, typing.Any]:
        return dict(
            epoch=self.epoch,
            early_stopped=self.early_stopped,
            early_stopped_max_epochs=self.early_stopped_max_epochs,
            train_batch=self.train_batch,
            train_total_batch=self.train_total_batch,
            val_batch=self.val_batch,
            val_total_batch=self.val_total_batch,
            last_train_loss=self.last_train_loss,
            last_train_accuracy=self.last_train_accuracy,
            last_train_metrics=self.last_train_metrics,
            last_val_loss=self.last_val_loss,
            last_val_accuracy=self.last_val_accuracy,
            last_val_metrics=self.last_val_metrics,
            last_loss=self.last_loss,
            last_accuracy=self.last_accuracy,
            last_metrics=self.last_metrics,
            best_epoch=self.best_epoch,
            best_accuracy=self.best_accuracy,
            best_train_accuracy=self.best_train_accuracy,
            best_val_accuracy=self.best_val_accuracy,
            total_train_time=self.total_train_time,
            total_val_time=self.total_val_time,
            accuracies=self.accuracies,
            losses=self.losses,
            times=self.times,
            metrics=self.metrics,
            val_accuracies=self.val_accuracies,
            val_losses=self.val_losses,
            val_times=self.val_times,
            val_metrics=self.val_metrics,
            last_epoch_accuracies=self.last_epoch_accuracies,
            last_epoch_losses=self.last_epoch_losses,
            last_epoch_times=self.last_epoch_times,
            last_epoch_metrics=self.last_epoch_metrics,
            last_epoch_val_accuracies=self.last_epoch_val_accuracies,
            last_epoch_val_losses=self.last_epoch_val_losses,
            last_epoch_val_times=self.last_epoch_val_times,
            last_epoch_val_metrics=self.last_epoch_val_metrics,
            epoch_cursor=self.epoch_cursor.state_dict() if self.epoch_cursor else None,
            batch_train_cursor=(
                self.batch_train_cursor.state_dict()
                if self.batch_train_cursor
                else None),
            batch_val_cursor=self.batch_val_cursor.state_dict() if self.batch_val_cursor else None,
        )

    @staticmethod
    def from_state_dict(state_props: dict[str, typing.Any]):
        epoch_cursor_value = state_props.get('epoch_cursor')
        epoch_cursor = ExecutionCursor.from_state_dict(
            epoch_cursor_value
        ) if epoch_cursor_value else None

        batch_train_cursor_value = state_props.get('batch_train_cursor')
        batch_train_cursor = ExecutionCursor.from_state_dict(
            batch_train_cursor_value
        ) if batch_train_cursor_value else None

        batch_val_cursor_value = state_props.get('batch_val_cursor')
        batch_val_cursor = ExecutionCursor.from_state_dict(
            batch_val_cursor_value
        ) if batch_val_cursor_value else None

        return TrainResult(
            epoch=state_props['epoch'],
            early_stopped=state_props['early_stopped'],
            early_stopped_max_epochs=state_props['early_stopped_max_epochs'],
            train_batch=state_props['train_batch'],
            train_total_batch=state_props['train_total_batch'],
            val_batch=state_props['val_batch'],
            val_total_batch=state_props['val_total_batch'],
            last_train_loss=state_props['last_train_loss'],
            last_train_accuracy=state_props['last_train_accuracy'],
            last_train_metrics=state_props['last_train_metrics'],
            last_val_loss=state_props['last_val_loss'],
            last_val_accuracy=state_props['last_val_accuracy'],
            last_val_metrics=state_props['last_val_metrics'],
            last_loss=state_props['last_loss'],
            last_accuracy=state_props['last_accuracy'],
            last_metrics=state_props['last_metrics'],
            best_epoch=state_props['best_epoch'],
            best_accuracy=state_props['best_accuracy'],
            best_train_accuracy=state_props['best_train_accuracy'],
            best_val_accuracy=state_props['best_val_accuracy'],
            total_train_time=state_props['total_train_time'],
            total_val_time=state_props['total_val_time'],
            accuracies=state_props['accuracies'],
            losses=state_props['losses'],
            times=state_props['times'],
            metrics=state_props['metrics'],
            val_accuracies=state_props['val_accuracies'],
            val_losses=state_props['val_losses'],
            val_times=state_props['val_times'],
            val_metrics=state_props['val_metrics'],
            last_epoch_accuracies=state_props['last_epoch_accuracies'],
            last_epoch_losses=state_props['last_epoch_losses'],
            last_epoch_times=state_props['last_epoch_times'],
            last_epoch_metrics=state_props['last_epoch_metrics'],
            last_epoch_val_accuracies=state_props['last_epoch_val_accuracies'],
            last_epoch_val_losses=state_props['last_epoch_val_losses'],
            last_epoch_val_times=state_props['last_epoch_val_times'],
            last_epoch_val_metrics=state_props['last_epoch_val_metrics'],
            epoch_cursor=epoch_cursor,
            batch_train_cursor=batch_train_cursor,
            batch_val_cursor=batch_val_cursor)

class TestResult(BaseResult):
    def __init__(
        self,
        epoch: int,
        batch: int,
        total_batch: int | None,
        accuracy: float,
        loss: float,
        total_time: int,
        last_epoch_accuracies: list[tuple[int, float]] | None = None,
        last_epoch_losses: list[tuple[int, float]] | None = None,
        last_epoch_times: list[tuple[int, int]] | None = None,
        last_epoch_metrics: list[tuple[int, typing.Any]] | None = None,
        batch_cursor: ExecutionCursor | None = None,
    ):
        self.epoch = epoch
        self.batch = batch
        self.total_batch = total_batch
        self.accuracy = accuracy
        self.loss = loss
        self.total_time = total_time
        self.last_epoch_accuracies = last_epoch_accuracies
        self.last_epoch_losses = last_epoch_losses
        self.last_epoch_times = last_epoch_times
        self.last_epoch_metrics = last_epoch_metrics
        self.batch_cursor = batch_cursor

    def state_dict(self) -> dict[str, typing.Any]:
        return dict(
            epoch=self.epoch,
            batch=self.batch,
            total_batch=self.total_batch,
            accuracy=self.accuracy,
            loss=self.loss,
            total_time=self.total_time,
            last_epoch_accuracies=self.last_epoch_accuracies,
            last_epoch_losses=self.last_epoch_losses,
            last_epoch_times=self.last_epoch_times,
            last_epoch_metrics=self.last_epoch_metrics,
            batch_cursor=self.batch_cursor.state_dict() if self.batch_cursor else None,
        )

    @staticmethod
    def from_state_dict(state_props: dict[str, typing.Any]):
        batch_cursor_value = state_props.get('batch_cursor')
        batch_cursor = ExecutionCursor.from_state_dict(
            batch_cursor_value
        ) if batch_cursor_value else None

        return TestResult(
            epoch=state_props['epoch'],
            batch=state_props['batch'],
            total_batch=state_props['total_batch'],
            accuracy=state_props['accuracy'],
            loss=state_props['loss'],
            total_time=state_props['total_time'],
            last_epoch_accuracies=state_props['last_epoch_accuracies'],
            last_epoch_losses=state_props['last_epoch_losses'],
            last_epoch_times=state_props['last_epoch_times'],
            batch_cursor=batch_cursor,
        )

####################################################
################## General States ##################
####################################################

OT = typing.TypeVar("OT", bound=optim.Optimizer)

class MinimalFullState(BaseResult):
    def __init__(
            self,
            train_results: TrainResult,
            test_results: TestResult | None):
        self.train_results = train_results
        self.test_results = test_results

    def state_dict(self) -> dict[str, typing.Any]:
        return dict(
            train_results=self.train_results.state_dict(),
            test_results=self.test_results.state_dict() if self.test_results else None)

    @staticmethod
    def from_minimal_state_dict(state_dict: dict[str, typing.Any]):
        train_results = TrainResult.from_state_dict(
            state_dict['train_results'])
        test_results = TestResult.from_state_dict(
            state_dict['test_results']) if state_dict['test_results'] else None

        return MinimalFullState(
            train_results=train_results,
            test_results=test_results)

class ModelMainState(typing.Generic[OT]):
    def __init__(
        self,
        model: nn.Module,
        optimizer: OT,
        scheduler: Scheduler | None = None,
        early_stopper: EarlyStopper | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopper = early_stopper

    def state_dict(self):
        return dict(
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict() if self.scheduler else None,
            early_stopper=self.early_stopper.state_dict() if self.early_stopper else None)

class ModelFullState(typing.Generic[OT], MinimalFullState):
    def __init__(
            self,
            model: nn.Module,
            optimizer: OT,
            best_state_dict: dict[str, typing.Any] | None,
            train_results: TrainResult,
            test_results: TestResult | None):
        super().__init__(
            train_results=train_results,
            test_results=test_results)
        self.model = model
        self.optimizer = optimizer
        self.best_state_dict = best_state_dict

    def state_dict(self):
        last_state_dict = ModelMainState(model=self.model, optimizer=self.optimizer).state_dict()
        best_state_dict = self.best_state_dict

        if (not best_state_dict) or (self.train_results.best_epoch == self.train_results.epoch):
            best_state_dict = last_state_dict

        return dict(
            best=best_state_dict,
            last=last_state_dict,
            train_results=self.train_results.state_dict(),
            test_results=self.test_results.state_dict() if self.test_results else None)

    @staticmethod
    def from_state_dict(
            model: nn.Module,
            optimizer: OT,
            use_best: bool,
            state_dict: dict[str, typing.Any]):
        best_state_dict = state_dict['best']
        last_state_dict = state_dict['last']
        outer_state_dict = best_state_dict if use_best else last_state_dict

        model.load_state_dict(outer_state_dict['model'])
        optimizer.load_state_dict(outer_state_dict['optimizer'])

        train_results = TrainResult.from_state_dict(
            state_dict['train_results'])
        test_results = TestResult.from_state_dict(
            state_dict['test_results']) if state_dict['test_results'] else None

        return ModelFullState(
            model=model,
            optimizer=optimizer,
            best_state_dict=best_state_dict,
            train_results=train_results,
            test_results=test_results)

####################################################
################## General Params ##################
####################################################

class MinimalHookParams():
    def __init__(
            self,
            current_amount: int,
            loss: float,
            accuracy: float):
        self.current_amount = current_amount
        self.loss = loss
        self.accuracy = accuracy

I = TypeVar("I")
O = TypeVar("O")
TRH = TypeVar("TRH", bound=MinimalHookParams)
TEH = TypeVar("TEH", bound=MinimalHookParams)

class MinimalEvalParams():
    def __init__(
            self,
            save_path: str | None,
            skip_load_state: bool):

        self.save_path = save_path
        self.skip_load_state = skip_load_state

class TrainBatchInfo(typing.Generic[M]):
    def __init__(
        self,
        loss: float | None,
        accuracy: float | None,
        metrics: M | None,
        count: int | None,
        batch: int | None,
        total_batch: int | None,
        first: bool | None,
        last: bool | None,
        start: float | None,
        prefix: str | None,
        validation: bool | None,
        test: bool | None
    ):
        self.loss = loss
        self.accuracy = accuracy
        self.metrics = metrics
        self.count = count
        self.batch = batch
        self.total_batch = total_batch
        self.first = first
        self.last = last
        self.start = start
        self.prefix = prefix
        self.validation = validation
        self.test = test

class TrainEpochInfo(typing.Generic[M]):
    def __init__(
        self,
        epochs: int | None,
        epoch: int | None,
        start_epoch: int | None,
        start: float | None,
        loss: float | None,
        val_loss: float | None,
        accuracy: float | None,
        val_accuracy: float | None,
        metrics: M | None,
        val_metrics: M | None,
        count: int | None,
        validate: bool,
        batch_interval: bool | None,
    ):
        self.epochs = epochs
        self.epoch = epoch
        self.start_epoch = start_epoch
        self.start = start
        self.loss = loss
        self.val_loss = val_loss
        self.accuracy = accuracy
        self.val_accuracy = val_accuracy
        self.metrics = metrics
        self.val_metrics = val_metrics
        self.count = count
        self.validate = validate
        self.batch_interval = batch_interval

class MinimalTestParams(MinimalEvalParams):
    def __init__(
            self,
            save_every: int | None,
            print_every: int | None,
            metric_every: int | None,
            early_stopper: EarlyStopper | None,
            get_batch_info: typing.Callable[[TrainBatchInfo], str] | None,
            save_path: str | None,
            skip_load_state: bool):

        super().__init__(
            save_path=save_path,
            skip_load_state=skip_load_state)

        self.save_every = save_every
        self.print_every = print_every
        self.metric_every = metric_every
        self.early_stopper = early_stopper
        self.get_batch_info = get_batch_info

class MinimalTrainParams(MinimalEvalParams, Generic[I, TRH, TEH]):
    def __init__(
            self,
            train_dataloader: DataLoader[I],
            validation_dataloader: DataLoader[I] | None,
            train_hook: typing.Callable[[TRH], None] | None,
            validation_hook: typing.Callable[[TEH], None] | None,
            epochs: int,
            batch_interval: bool | None,
            save_every: int | None,
            print_every: int | None,
            metric_every: int | None,
            early_stopper: EarlyStopper | None,
            save_path: str | None,
            skip_load_state: bool,
            get_epoch_info: typing.Callable[[TrainEpochInfo], str] | None,
            get_batch_info: typing.Callable[[TrainBatchInfo], str] | None):

        super().__init__(
            save_path=save_path,
            skip_load_state=skip_load_state)

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.train_hook = train_hook
        self.validation_hook = validation_hook
        self.epochs = epochs
        self.batch_interval = batch_interval
        self.save_every = save_every
        self.print_every = print_every
        self.metric_every = metric_every
        self.early_stopper = early_stopper
        self.get_epoch_info = get_epoch_info
        self.get_batch_info = get_batch_info

class BatchInOutParams(typing.Generic[I, O]):
    def __init__(
        self,
        input: I,
        full_output: O,
        output: torch.Tensor,
        target: torch.Tensor,
    ):
        self.input = input
        self.full_output = full_output
        self.output = output
        self.target = target

class SingleModelTrainParams(MinimalTrainParams[I, TRH, TEH], Generic[I, O, TRH, TEH]):
    def __init__(
            self,
            train_dataloader: DataLoader[I],
            validation_dataloader: DataLoader[I] | None,
            model: nn.Module,
            criterion: nn.Module | typing.Callable[[BatchInOutParams[I, O]], torch.Tensor],
            optimizer: optim.Optimizer,
            epochs: int,
            batch_interval: bool,
            save_every: int | None,
            print_every: int | None,
            metric_every: int | None,
            scheduler: Scheduler | None = None,
            step_only_on_accuracy_loss: bool = False,
            clip_grad_max: float | None = None,
            early_stopper: EarlyStopper | None = None,
            get_epoch_info: typing.Callable[[TrainEpochInfo], str] | None = None,
            get_batch_info: typing.Callable[[TrainBatchInfo], str] | None = None,
            train_hook: typing.Callable[[TRH], None] | None = None,
            validation_hook: typing.Callable[[TEH], None] | None = None,
            save_path: str | None = None,
            skip_load_state=False):
        super().__init__(
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            train_hook=train_hook,
            validation_hook=validation_hook,
            epochs=epochs,
            batch_interval=batch_interval,
            save_every=save_every,
            print_every=print_every,
            metric_every=metric_every,
            early_stopper=early_stopper,
            get_epoch_info=get_epoch_info,
            get_batch_info=get_batch_info,
            save_path=save_path,
            skip_load_state=skip_load_state)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.step_only_on_accuracy_loss = step_only_on_accuracy_loss
        self.clip_grad_max = clip_grad_max

class SingleModelMinimalEvalParams(MinimalEvalParams):
    def __init__(
        self,
        model: nn.Module,
        save_path: str | None = None,
        skip_load_state=False,
    ):
        super().__init__(
            save_path=save_path,
            skip_load_state=skip_load_state)

        self.model = model

class SingleModelTestParams(MinimalTestParams, Generic[I, O, TEH]):
    def __init__(
            self,
            model: nn.Module,
            dataloader: DataLoader[I],
            criterion: nn.Module | typing.Callable[[BatchInOutParams[I, O]], torch.Tensor],
            save_every: int | None,
            print_every: int | None,
            metric_every: int | None,
            early_stopper: EarlyStopper | None = None,
            get_batch_info: typing.Callable[[TrainBatchInfo], str] | None = None,
            hook: typing.Callable[[TEH], None] | None = None,
            save_path: str | None = None,
            skip_load_state=False):
        super().__init__(
            save_every=save_every,
            print_every=print_every,
            metric_every=metric_every,
            early_stopper=early_stopper,
            get_batch_info=get_batch_info,
            save_path=save_path,
            skip_load_state=skip_load_state)

        self.model = model
        self.dataloader = dataloader
        self.hook = hook
        self.criterion = criterion

ATR = TypeVar("ATR", bound=MinimalTrainParams)
ATE = TypeVar("ATE", bound=MinimalEvalParams)

####################################################
################## General States ##################
####################################################

class SingleModelEvalState(MinimalFullState):
    def __init__(
        self,
        model: nn.Module,
        train_results: TrainResult,
        test_results: TestResult | None,
        metrics: dict[str, typing.Any] | None,
    ):
        super().__init__(
            train_results=train_results,
            test_results=test_results)
        self.model = model
        self.metrics = metrics

    def state_dict(self):
        return dict(
            test_results=self.test_results.state_dict() if self.test_results else None,
            metrics=self.metrics)

    @staticmethod
    def from_state_dict(
        model: nn.Module,
        use_best: bool,
        state_dict: dict[str, typing.Any],
    ):
        best_state_dict = state_dict['best']
        last_state_dict = state_dict['last']
        outer_state_dict = best_state_dict if use_best else last_state_dict

        test_results_dict = state_dict.get('test_results')
        metrics = state_dict.get('metrics')

        model.load_state_dict(outer_state_dict['model'])
        train_results = TrainResult.from_state_dict(
            state_dict['train_results'])
        test_results = TestResult.from_state_dict(
            test_results_dict) if test_results_dict else None

        return SingleModelEvalState(
            model=model,
            train_results=train_results,
            test_results=test_results,
            metrics=metrics)

    @staticmethod
    def from_state_dict_with_params(
        params: SingleModelMinimalEvalParams,
        use_best: bool,
        state_dict: dict[str, typing.Any],
    ):
        return SingleModelEvalState.from_state_dict(
            model=params.model,
            use_best=use_best,
            state_dict=state_dict,
        )

class SingleModelFullState(MinimalFullState):
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        best_state_dict: dict[str, typing.Any] | None,
        train_results: TrainResult,
        test_results: TestResult | None,
        metrics: dict[str, typing.Any] | None,
        scheduler: Scheduler | None = None,
        early_stopper: EarlyStopper | None = None,
    ):
        super().__init__(
            train_results=train_results,
            test_results=test_results)
        self.model = model
        self.metrics = metrics
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopper = early_stopper
        self.best_state_dict = best_state_dict

    def state_dict(self) -> dict[str, typing.Any]:
        best_state_dict = self.best_state_dict
        last_state_dict = ModelMainState(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            early_stopper=self.early_stopper,
        ).state_dict()

        if (not best_state_dict) or (self.train_results.best_epoch == self.train_results.epoch):
            best_state_dict = last_state_dict

        return dict(
            best=best_state_dict,
            last=last_state_dict,
            train_results=self.train_results.state_dict(),
            test_results=self.test_results.state_dict() if self.test_results else None)

    @staticmethod
    def from_state_dict(
            model: nn.Module,
            optimizer: optim.Optimizer,
            scheduler: Scheduler | None,
            early_stopper: EarlyStopper | None,
            use_best: bool,
            state_dict: dict[str, typing.Any]):
        best_state_dict = state_dict['best']
        last_state_dict = state_dict['last']
        outer_state_dict = best_state_dict if use_best else last_state_dict

        model.load_state_dict(outer_state_dict['model'])
        optimizer.load_state_dict(outer_state_dict['optimizer'])

        scheduler_dict = outer_state_dict.get('scheduler')
        early_stopper_dict = outer_state_dict.get('early_stopper')

        if scheduler is not None and scheduler_dict is not None:
            scheduler.load_state_dict(scheduler_dict)

        if early_stopper is not None and early_stopper_dict is not None:
            early_stopper.load_state_dict(early_stopper_dict)

        test_results_dict = state_dict.get('test_results')
        metrics = state_dict.get('metrics')

        train_results = TrainResult.from_state_dict(
            state_dict['train_results'])
        test_results = TestResult.from_state_dict(
            test_results_dict) if test_results_dict else None

        return SingleModelFullState(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopper=early_stopper,
            train_results=train_results,
            test_results=test_results,
            metrics=metrics,
            best_state_dict=best_state_dict)

    @staticmethod
    def from_state_dict_with_params(
            params: SingleModelTrainParams[Any, Any, Any, Any],
            use_best: bool,
            state_dict: dict[str, typing.Any]):
        return SingleModelFullState.from_state_dict(
            model=params.model,
            optimizer=params.optimizer,
            scheduler=params.scheduler,
            early_stopper=params.early_stopper,
            use_best=use_best,
            state_dict=state_dict,
        )

####################################################
################## General Action ##################
####################################################

INF = typing.TypeVar("INF")

class AbstractAction(Generic[ATR, ATE, INF]):
    def __init__(
            self,
            train: typing.Callable[[ATR], TrainResult | None],
            test: typing.Callable[[ATE], TestResult | None],
            info: typing.Callable[[str], INF | None],
            load_eval_state: typing.Callable[[SingleModelMinimalEvalParams], None]):
        self.train = train
        self.test = test
        self.info = info
        self.load_eval_state = load_eval_state
