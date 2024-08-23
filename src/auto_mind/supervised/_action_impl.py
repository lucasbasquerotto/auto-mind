# ruff: noqa: E741 (ambiguous variable name)
# pylint: disable=too-many-lines
import os
import time
import math
import typing
import warnings
import torch
import torch.nn as nn
from torch import optim
from auto_mind.supervised._action import (
    BaseResult, MinimalEvalParams, MinimalFullState, MinimalTrainParams, SingleModelEvalState,
    SingleModelFullState, SingleModelMinimalEvalParams, SingleModelTestParams,
    SingleModelTrainParams, TestResult, TrainResult, MinimalTestParams, ExecutionCursor,
    EarlyStopper, TrainEarlyStopper, TrainEpochInfo, TrainBatchInfo)

I = typing.TypeVar("I")
O = typing.TypeVar("O")
M = typing.TypeVar("M", bound=nn.Module)
OT = typing.TypeVar("OT", bound=optim.Optimizer)
RV = typing.TypeVar("RV", bound=BaseResult)
MT = typing.TypeVar("MT")

S = typing.TypeVar("S", bound=BaseResult)

####################################################
################## State Handler ###################
####################################################

P = typing.TypeVar("P", bound=MinimalEvalParams)
S = typing.TypeVar("S", bound=BaseResult)

INF = typing.TypeVar("INF")
ATR = typing.TypeVar("ATR", bound=MinimalTrainParams)
ATE = typing.TypeVar("ATE", bound=MinimalTestParams)
STR = typing.TypeVar("STR", bound=MinimalFullState)
STE = typing.TypeVar("STE", bound=MinimalFullState)

class AbortedException(Exception):
    pass

class StateHandler(typing.Generic[INF, ATR, STR, ATE, STE]):
    def __init__(
        self,
        info_from_dict: typing.Callable[[dict[str, typing.Any]], INF],
        train_state_from_dict: typing.Callable[[ATR, dict[str, typing.Any]], STR | None] | None,
        new_train_state: typing.Callable[
            [ATR, TrainResult, dict[str, typing.Any] | None],
            STR | None
        ] | None,
        test_state_from_dict: typing.Callable[[
            ATE, dict[str, typing.Any]], STE | None] | None,
        new_test_state: typing.Callable[
            [ATE, TestResult, dict[str, typing.Any] | None],
            STE | None
        ] | None,
    ):
        self._info_from_dict = info_from_dict
        self._train_state_from_dict = train_state_from_dict
        self._new_train_state = new_train_state
        self._test_state_from_dict = test_state_from_dict
        self._new_test_state = new_test_state

    def _load_state(
        self,
        params: P,
        get_state: typing.Callable[[P, dict[str, typing.Any]], S | None] | None,
    ):
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
    ):
        if state and params.save_path:
            state_dict = state.state_dict()

            if last_state_dict:
                state_dict = last_state_dict | state_dict

            _save_state_dict(
                state_dict=state_dict,
                save_path=params.save_path)

    def _save_state_dict(self, save_path: str | None, state_dict: dict[str, typing.Any]):
        if state_dict and save_path:
            _save_state_dict(
                state_dict=state_dict,
                save_path=save_path)

    def info(self, save_path: str) -> INF | None:
        state_dict = _load_state_dict(save_path=save_path)
        info = self._info_from_dict(state_dict) if state_dict else None
        return info

    def load_train_state(self, params: ATR):
        return self._load_state(params, self._train_state_from_dict)

    def save_train_state(
        self,
        params: ATR,
        result: TrainResult,
        last_state_dict: dict[str, typing.Any] | None,
    ):
        new_train_state = self._new_train_state

        if new_train_state and params.save_path:
            state = new_train_state(params, result, last_state_dict)
            self._save_state(
                params=params,
                state=state,
                last_state_dict=last_state_dict)

    def load_test_state(self, params: ATE):
        return self._load_state(params, self._test_state_from_dict)

    def save_test_state(
        self,
        params: ATE,
        result: TestResult,
        last_state_dict: dict[str, typing.Any] | None,
    ):
        new_test_state = self._new_test_state

        if new_test_state and params.save_path:
            state = new_test_state(params, result, last_state_dict)
            self._save_state(
                params=params,
                state=state,
                last_state_dict=last_state_dict)


SMTR = typing.TypeVar("SMTR", bound=SingleModelTrainParams)
SMTE = typing.TypeVar("SMTE", bound=SingleModelTestParams)

class MinimalStateWithMetrics(MinimalFullState):
    def __init__(
            self,
            completed: bool,
            train_results: TrainResult,
            test_results: TestResult | None,
            metrics: dict[str, typing.Any] | None):
        super().__init__(
            train_results=train_results,
            test_results=test_results)
        self.metrics = metrics
        self.completed = completed

    @staticmethod
    def from_state_dict(state_dict: dict[str, typing.Any]):
        completed = bool(state_dict.get('completed'))
        train_results = TrainResult.from_state_dict(
            state_dict['train_results'])
        test_results = TestResult.from_state_dict(
            state_dict['test_results']) if state_dict['test_results'] else None
        metrics = state_dict.get('metrics')

        return MinimalStateWithMetrics(
            completed=completed,
            train_results=train_results,
            test_results=test_results,
            metrics=metrics)

class SingleModelStateHandler(StateHandler[
            MinimalStateWithMetrics,
            SMTR,
            SingleModelFullState,
            SMTE,
            SingleModelEvalState
        ],
        typing.Generic[SMTR, SMTE]):
    def __init__(self, use_best: bool):
        def get_eval_state(params: SingleModelMinimalEvalParams, state_dict: dict[str, typing.Any]):
            return SingleModelEvalState.from_state_dict_with_params(
                params,
                use_best=use_best,
                state_dict=state_dict)

        def info_from_dict(state_dict: dict[str, typing.Any]):
            return MinimalStateWithMetrics.from_state_dict(state_dict)

        def train_state_from_dict(params: SMTR, state_dict: dict[str, typing.Any]):
            return SingleModelFullState.from_state_dict_with_params(
                params,
                use_best=False,
                state_dict=state_dict)

        def new_train_state(
            params: SMTR,
            train_results: TrainResult,
            last_state_dict: dict[str, typing.Any] | None,
        ):
            return SingleModelFullState(
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
            params: SMTE,
            test_results: TestResult,
            last_state_dict: dict[str, typing.Any] | None,
        ):
            return SingleModelEvalState(
                model=params.model,
                train_results=TrainResult.from_state_dict(
                    last_state_dict['train_results']),
                test_results=test_results,
                metrics=last_state_dict.get('metrics'),
            ) if last_state_dict and last_state_dict.get('train_results') else None

        super().__init__(
            info_from_dict=info_from_dict,
            train_state_from_dict=train_state_from_dict,
            new_train_state=new_train_state,
            test_state_from_dict=get_eval_state,
            new_test_state=new_test_state,
        )

        self.get_eval_state = get_eval_state

    def load_eval_state(self, params: SingleModelMinimalEvalParams):
        return self._load_state(params, self.get_eval_state)

    def load_state_with_metrics(self, save_path: str) -> MinimalStateWithMetrics | None:
        state_dict = _load_state_dict(save_path=save_path)
        return MinimalStateWithMetrics.from_state_dict(state_dict) if state_dict else None

    def save_metrics(self, metrics: dict[str, typing.Any], save_path: str | None):
        if save_path:
            last_state_dict = _load_state_dict(save_path=save_path)

            if last_state_dict:
                last_state_dict['metrics'] = metrics
                self._save_state_dict(save_path, last_state_dict)

    def define_as_completed(self, completed: bool, save_path: str | None):
        if save_path:
            last_state_dict = _load_state_dict(save_path=save_path)

            if last_state_dict:
                last_state_dict['completed'] = completed
                self._save_state_dict(save_path, last_state_dict)

####################################################
################## Batch Handlers ##################
####################################################

class BatchHandlerRunParams(typing.Generic[I]):
    def __init__(self, data: I, batch: int, amount: int):
        self.data = data
        self.batch = batch
        self.amount = amount

class BatchHandlerData(typing.Generic[I, O]):
    def __init__(
        self,
        amount: int,
        loss: float,
        accuracy: float,
        input: I, # pylint: disable=redefined-builtin
        full_output: O,
        output: torch.Tensor,
        target: torch.Tensor,
    ):
        self.amount = amount
        self.loss = loss
        self.accuracy = accuracy
        self.input = input
        self.full_output = full_output
        self.output = output
        self.target = target

class BatchHandlerResult:
    def __init__(
        self,
        total_loss: float,
        total_accuracy: float,
        total_time: int,
        total_metrics: typing.Any | None,
    ):
        self.total_loss = total_loss
        self.total_accuracy = total_accuracy
        self.total_time = total_time
        self.total_metrics = total_metrics

class MetricsHandlerInput(BatchHandlerData[I, O], typing.Generic[I, O]):
    def __init__(self, out: BatchHandlerData[I, O], time_diff: int):
        super().__init__(
            amount=out.amount,
            loss=out.loss,
            accuracy=out.accuracy,
            input=out.input,
            full_output=out.full_output,
            output=out.output,
            target=out.target)

        self.time_diff = time_diff

class MetricsHandler(typing.Generic[I, O, MT]):
    def define(self, data: MetricsHandlerInput[I, O]) -> MT:
        raise NotImplementedError

    def add(self, current: MT | None, metrics: MT):
        raise NotImplementedError

class TensorMetricsHandler(MetricsHandler[torch.Tensor, torch.Tensor, MT], typing.Generic[MT]):
    def define(self, data: MetricsHandlerInput[torch.Tensor, torch.Tensor]) -> MT:
        raise NotImplementedError

    def add(self, current: MT | None, metrics: MT):
        raise NotImplementedError

class BatchHandler():
    def __init__(
        self,
        cursor: ExecutionCursor | None,
        metrics_handler: MetricsHandler | None,
        best_accuracy: float | None,
    ):
        self.amount = cursor.amount if cursor else 0
        self.total_loss = cursor.total_loss if cursor else 0.0
        self.total_accuracy = cursor.total_accuracy if cursor else 0.0
        self.total_time: int = cursor.total_time if cursor else 0
        self.total_metrics = cursor.total_metrics if cursor else None
        self.metrics_handler = metrics_handler
        self.best_accuracy = best_accuracy

    def verify_early_stop(self):
        """
        Raises exception if the run should be aborted.
        """

    def skip(self, batch: int) -> bool: # pylint: disable=unused-argument
        return True

    def handle(
        self,
        batch: int,
        total_batch: int | None,
        amount: int,
        last: bool,
        loss: float,
        accuracy: float,
        time_diff: int,
        batch_metrics: typing.Any | None,
    ):
        raise NotImplementedError()

    def handle_main(
        self,
        amount: int,
        loss: float,
        accuracy: float,
        time_diff: int,
        batch_metrics: typing.Any | None,
    ):
        self.amount += amount
        self.total_loss += loss
        self.total_accuracy += accuracy * amount
        self.total_time += time_diff

        if self.metrics_handler and batch_metrics:
            self.total_metrics = self.metrics_handler.add(self.total_metrics, batch_metrics)

    def run(
        self,
        dataloader: typing.Iterable[I],
        fn: typing.Callable[[BatchHandlerRunParams[I]], BatchHandlerData],
        epoch: int,
        random_seed: int | None,
    ) -> BatchHandlerResult:
        self.verify_early_stop()

        random_seed = (random_seed or 1) * (epoch + 1)
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
        out: BatchHandlerData[I, typing.Any] | None = None

        metrics_handler = self.metrics_handler

        def handle(last: bool):
            nonlocal start_time
            nonlocal batch_metrics

            assert out is not None

            loss_value = out.loss
            batch_accuracy = out.accuracy

            end_time = time.time()
            time_diff = time_diff_millis(start_time, end_time)
            start_time = end_time

            if metrics_handler:
                metrics_params = MetricsHandlerInput(
                    out=out,
                    time_diff=time_diff)
                batch_metrics = metrics_handler.define(metrics_params)

            with torch.no_grad():
                self.handle(
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
                handle(last=False)

            self.verify_early_stop()
            batch += 1

            if self.skip(batch):
                continue

            out = fn(BatchHandlerRunParams(
                data=data,
                batch=batch,
                amount=self.amount))

            current_amount = out.amount

            if not current_amount:
                break

        amount = current_amount + self.amount

        if not amount:
            raise Exception('The dataloader is empty')

        if out is not None:
            handle(last=True)

        total_loss = self.total_loss / self.amount
        total_accuracy = self.total_accuracy / self.amount

        return BatchHandlerResult(
            total_loss=total_loss,
            total_accuracy=total_accuracy,
            total_time=self.total_time,
            total_metrics=self.total_metrics)

class GeneralBatchHandlerParams:
    def __init__(
        self,
        save_every: int | None = None,
        print_every: int | None = None,
        metric_every: int | None = None,
    ):
        self.save_every = save_every
        self.print_every = print_every
        self.metric_every = metric_every

class GeneralBatchHandlerResults:
    def __init__(
        self,
        batch: int,
        total_batch: int | None,
        cursor: ExecutionCursor | None,
        last_epoch_accuracies: list[tuple[int, float]] | None,
        last_epoch_losses: list[tuple[int, float]] | None,
        last_epoch_times: list[tuple[int, int]] | None,
        last_epoch_metrics: list[tuple[int, typing.Any]] | None,
    ):
        self.batch = batch
        self.total_batch = total_batch
        self.cursor = cursor
        self.last_epoch_accuracies = last_epoch_accuracies
        self.last_epoch_losses = last_epoch_losses
        self.last_epoch_times = last_epoch_times
        self.last_epoch_metrics = last_epoch_metrics

def default_batch_info(info: TrainBatchInfo):
    print_loss = info.loss or 0
    print_accuracy = info.accuracy or 0
    print_count = info.count or 1
    batch = info.batch or 0
    total_batch = info.total_batch
    first = info.first
    last = info.last
    start = info.start or time.time()
    print_prefix = info.prefix or ''
    validation = info.validation
    test = info.test

    loss_avg = f'{(print_loss / print_count):.4f}'
    loss_avg = f'[loss: {loss_avg}]'

    acc_str = f'{100.0 * print_accuracy / print_count:>5.1f}%'
    acc_str = f'[accuracy: {acc_str}]'

    batch_cap = 10 if total_batch is None else math.ceil(math.log10(total_batch))
    batch_main_str = f'{batch:>{batch_cap}}'
    batch_str = f'[batch: {batch_main_str}]' if total_batch is None else (
        f'[batch: {batch_main_str}/{total_batch}]')
    now = time.time()
    diff_time = now - start
    time_str = _as_minutes(diff_time, spaces=11)
    time_str = f'[time: {time_str}]'

    result = f'{print_prefix}{batch_str} {time_str} {acc_str} {loss_avg}'

    if (first and validation) or (last and test):
        result_len = len(result)
        separator = ('=' if test else '-') * result_len
        result = f'{separator}\n{result}' if first else f'{result}\n{separator}'

    return result

def default_epoch_info(info: TrainEpochInfo):
    epochs = info.epochs or 0
    epoch = info.epoch or 0
    start_epoch = info.start_epoch or 0
    start = info.start or time.time()
    loss = info.loss or 0
    val_loss = info.val_loss or 0
    accuracy = info.accuracy or 0
    val_accuracy = info.val_accuracy or 0
    count = info.count or 1
    validate = info.validate
    batch_interval = info.batch_interval

    train_loss_avg = loss / count
    val_loss_avg = val_loss / count
    loss_avg = (
        f'[val_loss: {val_loss_avg:.4f}, train_loss: {train_loss_avg:.4f}]'
        if validate
        else f'[loss: {train_loss_avg:.4f}]')

    train_acc_str = f'{100.0 * accuracy / count:>5.1f}%'
    val_acc_str = f'{100.0 * val_accuracy / count:>5.1f}%'
    acc_str = (
        f'[val_accuracy: {val_acc_str}, train_accuracy: {train_acc_str}]'
        if validate
        else f'[accuracy: {train_acc_str}]')

    epoch_cap = math.ceil(math.log10(epochs))
    epoch_str = f'[end of epoch {epoch:>{epoch_cap}} ({(100.0 * epoch / epochs):>5.1f}%)]'
    time_str = _time_since(start, (epoch - start_epoch + 1) / (epochs - start_epoch + 1), spaces=11)
    time_str = f'[time: {time_str}]'
    result = f'{epoch_str} {time_str} {acc_str} {loss_avg}'

    if batch_interval:
        result_len = len(result)
        separator_b = '-' * result_len
        separator_e = '=' * result_len
        result = f'{separator_b}\n{result}\n{separator_e}'

    return result

class GeneralBatchHandler(BatchHandler):
    def __init__(
        self,
        params: GeneralBatchHandlerParams,
        get_results: typing.Callable[[], GeneralBatchHandlerResults],
        update_results: typing.Callable[[GeneralBatchHandlerResults], None],
        print_prefix: str,
        get_batch_info: typing.Callable[[TrainBatchInfo], str],
        save_state: typing.Callable[[], None],
        metrics_handler: MetricsHandler | None,
        early_stopper: EarlyStopper | None = None,
        validation: bool = False,
        test: bool = False,
        best_accuracy: float | None = None,
    ):
        results = get_results()
        cursor = results.cursor

        super().__init__(
            cursor=cursor,
            metrics_handler=metrics_handler,
            best_accuracy=best_accuracy)

        self._params = params
        self._get_results = get_results
        self._update_results = update_results
        self._print_prefix = print_prefix
        self._save_state = save_state
        self._early_stopper = early_stopper
        self._get_batch_info = get_batch_info
        self._validation = validation
        self._test = test

        self._print_loss = 0
        self._print_accuracy = 0
        self._print_count = 0
        self._print_metrics: typing.Any | None = None

        self._metric_loss = 0
        self._metric_accuracy = 0
        self._metric_count = 0
        self._metric_time = 0
        self._metrics: typing.Any | None = None

        self._start = time.time()

    def verify_early_stop(self):
        if not self._early_stopper:
            return

        if self._early_stopper.check():
            raise AbortedException()

    def skip(self, batch: int) -> bool:
        results = self._get_results()
        start_batch = results.batch + 1
        return batch < start_batch

    def handle(
        self,
        batch: int,
        total_batch: int | None,
        amount: int,
        last: bool,
        loss: float,
        accuracy: float,
        time_diff: int,
        batch_metrics: typing.Any | None,
    ):
        if not self.skip(batch):
            self._handle(
                batch=batch,
                total_batch=total_batch,
                amount=amount,
                last=last,
                loss=loss,
                accuracy=accuracy,
                time_diff=time_diff,
                batch_metrics=batch_metrics,
            )

    def _handle(
        self,
        batch: int,
        total_batch: int | None,
        amount: int,
        last: bool,
        loss: float,
        accuracy: float,
        time_diff: int,
        batch_metrics: typing.Any | None,
    ):
        super().handle_main(
            amount=amount,
            loss=loss,
            accuracy=accuracy,
            time_diff=time_diff,
            batch_metrics=batch_metrics,
        )

        params = self._params
        results = self._get_results()
        update_results = self._update_results
        print_prefix = self._print_prefix
        save_state = self._save_state
        metrics_handler = self.metrics_handler

        self._print_loss += loss / amount
        self._print_accuracy += accuracy
        self._print_count += 1

        print_loss = self._print_loss
        print_accuracy = self._print_accuracy
        print_count = self._print_count

        self._metric_loss += loss / amount
        self._metric_accuracy += accuracy
        self._metric_time += time_diff
        self._metric_count += 1

        metric_loss = self._metric_loss
        metric_accuracy = self._metric_accuracy
        metric_time = self._metric_time
        metric_count = self._metric_count

        if metrics_handler:
            self._print_metrics = metrics_handler.add(self._print_metrics, batch_metrics)
            self._metrics = metrics_handler.add(self._metrics, batch_metrics)
        else:
            self._print_metrics = None
            self._metrics = None

        metrics = self._metrics
        print_metrics = self._print_metrics

        start = self._start

        save_every = params.save_every
        print_every = params.print_every
        metric_every = params.metric_every

        # print every print_every batches, but not on the last batch (it should be printed outside)
        do_print = (print_every is not None) and (last or (batch % print_every == 0))
        # save every save_every batches, but not on the last batch (it should be saved outside)
        do_save = (save_every is not None) and (last or (batch % save_every == 0))
        # update the persistent result, which is done when going to save,
        # and also when defining the metrics
        do_update = do_save or last or (metric_every is None) or (batch % metric_every == 0)
        # change metric values when updating, as long as metric_every is set
        do_metric = do_update and (metric_every is not None)

        if do_print:
            info = TrainBatchInfo(
                loss=print_loss,
                accuracy=print_accuracy,
                metrics=print_metrics,
                count=print_count,
                batch=batch,
                total_batch=total_batch,
                first=batch <= print_every if print_every is not None else False,
                last=last,
                start=start,
                prefix=print_prefix,
                validation=self._validation,
                test=self._test)
            info_str = self._get_batch_info(info)

            if info_str:
                print(info_str)

            self._print_count = 0
            self._print_loss = 0
            self._print_accuracy = 0
            self._print_metrics = None

        if do_update:
            results.batch = batch
            results.total_batch = total_batch
            results.cursor = ExecutionCursor(
                amount=self.amount,
                total_loss=self.total_loss,
                total_accuracy=self.total_accuracy,
                total_metrics=self.total_metrics,
                total_time=self.total_time)

            if do_metric:
                accuracies = results.last_epoch_accuracies or []
                losses = results.last_epoch_losses or []
                times = results.last_epoch_times or []
                epoch_metrics = results.last_epoch_metrics or []

                accuracies.append((batch, metric_accuracy / metric_count))
                losses.append((batch, metric_loss / metric_count))
                times.append((batch, metric_time))
                epoch_metrics.append((batch, metrics))

                results.last_epoch_accuracies = accuracies
                results.last_epoch_losses = losses
                results.last_epoch_times = times
                results.last_epoch_metrics = epoch_metrics

            update_results(results)

            self._metric_count = 0
            self._metric_loss = 0
            self._metric_accuracy = 0
            self._metrics = None

            if do_save:
                save_state()

class TrainBatchHandler(GeneralBatchHandler, typing.Generic[ATR]):
    def __init__(
        self,
        validation: bool,
        epoch: int,
        params: ATR,
        results: TrainResult,
        get_batch_info: typing.Callable[[TrainBatchInfo], str],
        save_state: typing.Callable[[TrainResult], None],
        metrics_handler: MetricsHandler | None,
        early_stopper: EarlyStopper | None = None,
    ):
        def get_results():
            batch = (
                results.val_batch
                if validation
                else results.train_batch)

            total_batch = (
                results.val_total_batch
                if validation
                else results.train_total_batch)

            cursor = (
                results.batch_val_cursor
                if validation
                else results.batch_train_cursor)

            accuracies = (
                results.last_epoch_val_accuracies
                if validation
                else results.last_epoch_accuracies)

            losses = (
                results.last_epoch_val_losses
                if validation
                else results.last_epoch_losses)

            times = (
                results.last_epoch_val_times
                if validation
                else results.last_epoch_times)

            metrics = (
                results.last_epoch_val_metrics
                if validation
                else results.last_epoch_metrics)

            return GeneralBatchHandlerResults(
                batch=batch,
                total_batch=total_batch,
                cursor=cursor,
                last_epoch_accuracies=accuracies,
                last_epoch_losses=losses,
                last_epoch_times=times,
                last_epoch_metrics=metrics)

        def update_results(main_result: GeneralBatchHandlerResults):
            accuracies = main_result.last_epoch_accuracies
            losses = main_result.last_epoch_losses
            times = main_result.last_epoch_times
            metrics = main_result.last_epoch_metrics

            if validation:
                results.val_batch = main_result.batch
                results.val_total_batch = main_result.total_batch
                results.batch_val_cursor = main_result.cursor
                results.last_epoch_val_accuracies = accuracies
                results.last_epoch_val_losses = losses
                results.last_epoch_val_times = times
                results.last_epoch_val_metrics = metrics
            else:
                results.train_batch = main_result.batch
                results.train_total_batch = main_result.total_batch
                results.batch_train_cursor = main_result.cursor
                results.last_epoch_accuracies = accuracies
                results.last_epoch_losses = losses
                results.last_epoch_times = times
                results.last_epoch_metrics = metrics

        epochs = params.epochs
        epoch_cap = math.ceil(math.log10(epoch))
        epoch_str = f'{epoch:>{epoch_cap}} ({100.0 * epoch / epochs:>5.1f}%)'
        type_str = 'validation' if validation else 'train'
        print_prefix = f'> [{type_str}] [epoch {epoch_str}] '

        super().__init__(
            params=GeneralBatchHandlerParams(
                save_every=params.save_every,
                print_every=params.print_every,
                metric_every=params.metric_every),
            get_results=get_results,
            update_results=update_results,
            print_prefix=print_prefix,
            get_batch_info=get_batch_info,
            save_state=lambda: save_state(results),
            early_stopper=early_stopper,
            metrics_handler=metrics_handler,
            validation=validation,
            best_accuracy=results.best_accuracy if results else None)

        self._active = params.batch_interval
        self.best_accuracy = results.best_accuracy if results else None

    def verify_early_stop(self):
        if self._active:
            super().verify_early_stop()

    def skip(self, batch: int) -> bool:
        if self._active:
            return super().skip(batch)
        return False

    def handle(
        self,
        batch: int,
        total_batch: int | None,
        amount: int,
        last: bool,
        loss: float,
        accuracy: float,
        time_diff: int,
        batch_metrics: typing.Any | None,
    ):
        if self._active:
            super().handle(
                batch=batch,
                total_batch=total_batch,
                amount=amount,
                last=last,
                loss=loss,
                accuracy=accuracy,
                time_diff=time_diff,
                batch_metrics=batch_metrics)
        else:
            super().handle_main(
                amount=amount,
                loss=loss,
                accuracy=accuracy,
                time_diff=time_diff,
                batch_metrics=batch_metrics)

class TestBatchHandler(GeneralBatchHandler, typing.Generic[ATE]):
    def __init__(
        self,
        params: ATE,
        results: TestResult,
        get_batch_info: typing.Callable[[TrainBatchInfo], str],
        save_state: typing.Callable[[TestResult], None],
        metrics_handler: MetricsHandler | None,
        early_stopper: EarlyStopper | None = None,
    ):
        def get_results():
            return GeneralBatchHandlerResults(
                batch=results.batch,
                total_batch=results.total_batch,
                cursor=results.batch_cursor,
                last_epoch_accuracies=results.last_epoch_accuracies,
                last_epoch_losses=results.last_epoch_losses,
                last_epoch_times=results.last_epoch_times,
                last_epoch_metrics=results.last_epoch_metrics)

        def update_results(main_result: GeneralBatchHandlerResults):
            nonlocal results
            results.batch = main_result.batch
            results.total_batch = main_result.total_batch
            results.batch_cursor = main_result.cursor
            results.last_epoch_accuracies = main_result.last_epoch_accuracies
            results.last_epoch_losses = main_result.last_epoch_losses
            results.last_epoch_times = main_result.last_epoch_times

        super().__init__(
            params=GeneralBatchHandlerParams(
                save_every=params.save_every,
                print_every=params.print_every,
                metric_every=params.metric_every),
            get_results=get_results,
            update_results=update_results,
            print_prefix='[test] ',
            get_batch_info=get_batch_info,
            save_state=lambda: save_state(results),
            early_stopper=early_stopper,
            metrics_handler=metrics_handler,
            test=True)

####################################################
################## Action Wrapper ##################
####################################################

AWP = typing.TypeVar("AWP")
AWS = typing.TypeVar("AWS")

class ActionWrapperActionParams(typing.Generic[AWP, AWS]):
    def __init__(
        self,
        main_params: AWP,
        full_state: AWS | None,
        epoch: int,
        batch_handler: BatchHandler,
    ):
        self.main_params = main_params
        self.full_state = full_state
        self.epoch = epoch
        self.batch_handler = batch_handler

class ActionWrapper(typing.Generic[INF, ATR, ATE]):
    def __init__(
        self,
        state_handler: StateHandler[INF, ATR, STR, ATE, STE],
        train_epoch: typing.Callable[[ActionWrapperActionParams[ATR, STR]], BatchHandlerResult],
        validate: typing.Callable[
            [ActionWrapperActionParams[ATR, STR]],
            BatchHandlerResult | None
        ] | None,
        test_inner: typing.Callable[
            [ActionWrapperActionParams[ATE, STE]],
            BatchHandlerResult | None
        ] | None,
        metrics_handler: MetricsHandler | None,
    ):
        self.state_handler = state_handler
        self.train_epoch = train_epoch
        self.validate = validate
        self.test_inner = test_inner
        self.metrics_handler = metrics_handler

    def can_run(self, early_stopper: EarlyStopper | None) -> bool:
        if not early_stopper:
            return True

        return not early_stopper.check()

    def _new_train_result(self):
        return TrainResult(
            epoch=0,
            early_stopped=False,
            early_stopped_max_epochs=0,
            train_batch=0,
            train_total_batch=None,
            val_batch=0,
            val_total_batch=None,
            last_loss=0.0,
            last_accuracy=0.0,
            last_metrics=None,
            last_train_loss=0.0,
            last_train_accuracy=0.0,
            last_train_metrics=None,
            last_val_loss=0.0,
            last_val_accuracy=0.0,
            last_val_metrics=None,
            best_epoch=0,
            best_accuracy=0.0,
            best_train_accuracy=0.0,
            best_val_accuracy=0.0,
            total_train_time=0,
            total_val_time=0,
            accuracies=[],
            losses=[],
            times=[],
            metrics=[] if self.metrics_handler else None,
            val_accuracies=[] if self.validate else None,
            val_losses=[] if self.validate else None,
            val_times=[] if self.validate else None,
            val_metrics=[] if self.validate and self.metrics_handler else None)

    def train(self, params: ATR) -> TrainResult | None:
        full_state, _ = self.state_handler.load_train_state(params)
        results: TrainResult = full_state.train_results if full_state else self._new_train_result()

        try:
            if (not results) or (not results.early_stopped):
                return self._train(params)

            return results
        except AbortedException:
            return None

    def test(self, params: ATE) -> TestResult | None:
        try:
            return self._test(params)
        except AbortedException:
            return None

    def _train(self, params: ATR) -> TrainResult:
        epochs = params.epochs
        batch_interval = params.batch_interval
        save_every = params.save_every
        print_every = params.print_every
        metric_every = params.metric_every
        early_stopper = params.early_stopper
        get_epoch_info = params.get_epoch_info or default_epoch_info

        full_state, state_dict = self.state_handler.load_train_state(params)
        results: TrainResult = full_state.train_results if full_state else self._new_train_result()
        start_epoch = results.epoch + 1

        print_count = 0
        print_loss = 0
        print_accuracy = 0
        print_val_loss = 0
        print_val_accuracy = 0

        # Keep track of losses and accuracies for the metrics
        metric_count = 0
        metric_loss = 0
        metric_accuracy = 0
        metric_train_time = 0
        metric_val_loss = 0
        metric_val_accuracy = 0
        metric_val_time = 0

        print_metrics: typing.Any | None = None
        print_val_metrics: typing.Any | None = None
        metrics: typing.Any | None = None

        epoch = 0
        train_loss, train_accuracy = 0.0, 0.0
        val_loss, val_accuracy = 0.0, 0.0
        train_metrics: typing.Any | None = None
        val_metrics: typing.Any | None = None

        get_batch_info = params.get_batch_info if params.get_batch_info else default_batch_info
        metrics_handler = self.metrics_handler

        def update_results():
            results.epoch = epoch
            results.train_batch = 0
            results.val_batch = 0
            results.batch_train_cursor = None
            results.batch_val_cursor = None
            results.epoch_cursor = ExecutionCursor(
                amount=metric_count,
                total_loss=metric_loss,
                total_accuracy=metric_accuracy,
                total_time=metric_train_time,
                total_metrics=metrics)
            results.last_train_loss = train_loss
            results.last_train_accuracy = train_accuracy
            results.last_train_metrics = train_metrics
            results.last_val_loss = val_loss
            results.last_val_accuracy = val_accuracy
            results.last_val_metrics = val_metrics

            if not self.validate:
                results.last_loss = train_loss
                results.last_accuracy = train_accuracy
                results.last_metrics = train_metrics
            else:
                results.last_loss = val_loss
                results.last_accuracy = val_accuracy
                results.last_metrics = val_metrics

        start = time.time()

        if print_every is not None:
            if start_epoch < epochs + 1:
                print_str = f'Starting training for {epochs} epochs...'
                max_len = len(print_str)

                if start_epoch > 1:
                    next_print_str = f'(starting from epoch {start_epoch})'
                    print_str += f'\n{next_print_str}'
                    max_len = max(max_len, len(next_print_str))

                separator = '=' * max_len
                print(print_str)
                print(separator)
            else:
                print(f'Training already completed ({epochs} epochs).')

        def verify_early_stop(force_save: bool):
            early_stopper = params.early_stopper

            if early_stopper:
                stopped = early_stopper.check()
                finished = isinstance(
                    early_stopper,
                    TrainEarlyStopper,
                ) and early_stopper.check_finish()

                if finished or stopped:
                    save_result = (
                        (finished and (not results.early_stopped))
                        or
                        (
                            force_save
                            and
                            stopped
                            and
                            (epoch > start_epoch)
                            and
                            (epoch > results.epoch)
                        )
                    )

                    if save_result:
                        update_results()

                        if finished:
                            results.early_stopped = True
                            results.early_stopped_max_epochs = epochs

                        self.state_handler.save_train_state(
                            params,
                            results,
                            state_dict)

                    if finished:
                        return results
                    else:
                        raise AbortedException()

            return None

        try:
            for epoch in range(start_epoch, epochs + 1):
                r = verify_early_stop(force_save=True)
                if r:
                    return r

                batch_handler = TrainBatchHandler(
                    validation=False,
                    epoch=epoch,
                    params=params,
                    results=results,
                    save_state=lambda result: self.state_handler.save_train_state(
                        params, result, state_dict),
                    early_stopper=early_stopper,
                    metrics_handler=metrics_handler,
                    get_batch_info=get_batch_info)

                train_params = ActionWrapperActionParams(
                    main_params=params,
                    full_state=full_state,
                    epoch=epoch,
                    batch_handler=batch_handler)

                with torch.set_grad_enabled(True):
                    result = self.train_epoch(train_params)

                train_loss = result.total_loss
                train_accuracy = result.total_accuracy
                train_time = result.total_time
                train_metrics = result.total_metrics

                metric_loss += train_loss
                print_loss += train_loss
                metric_accuracy += train_accuracy
                print_accuracy += train_accuracy
                metric_train_time += train_time

                if train_metrics:
                    print_metrics = metrics_handler.add(
                        print_metrics,
                        train_metrics,
                    ) if metrics_handler else None
                    metrics = metrics_handler.add(
                        metrics,
                        train_metrics,
                    ) if metrics_handler else None

                val_result = None

                if not self.validate:
                    if params.early_stopper and isinstance(params.early_stopper, TrainEarlyStopper):
                        params.early_stopper.update_epoch(
                            loss=train_loss,
                            accuracy=train_accuracy,
                            metrics=train_metrics)
                else:
                    batch_handler = TrainBatchHandler(
                        validation=True,
                        epoch=epoch,
                        params=params,
                        results=results,
                        save_state=lambda result: self.state_handler.save_train_state(
                            params, result, state_dict),
                        early_stopper=early_stopper,
                        metrics_handler=metrics_handler,
                        get_batch_info=get_batch_info)

                    val_params = ActionWrapperActionParams(
                        main_params=params,
                        full_state=full_state,
                        epoch=epoch,
                        batch_handler=batch_handler)

                    with torch.set_grad_enabled(False):
                        val_result = self.validate(val_params)

                    if val_result:
                        val_loss = val_result.total_loss
                        val_accuracy = val_result.total_accuracy
                        val_time = val_result.total_time
                        val_metrics_item = val_result.total_metrics

                        metric_val_loss += val_loss
                        print_val_loss += val_loss
                        metric_val_accuracy += val_accuracy
                        print_val_accuracy += val_accuracy
                        metric_val_time += val_time

                        if val_metrics_item:
                            print_val_metrics = metrics_handler.add(
                                print_val_metrics,
                                val_metrics_item,
                            ) if metrics_handler else None
                            val_metrics = metrics_handler.add(
                                val_metrics,
                                val_metrics_item,
                            ) if metrics_handler else None

                        if params.early_stopper and isinstance(
                            params.early_stopper,
                            TrainEarlyStopper,
                        ):
                            params.early_stopper.update_epoch(
                                loss=val_loss,
                                accuracy=val_accuracy,
                                metrics=val_metrics_item)

                print_count += 1
                metric_count += 1
                last = epoch == epochs

                # print every print_every epochs, or if last epoch,
                # or if batch_interval is set (to print partial epochs)
                do_print = (
                    (print_every is not None)
                    and
                    (batch_interval or last or (epoch % print_every == 0))
                )
                # save every save_every epochs, or if last epoch,
                # or if batch_interval is set (to save partial epochs)
                do_save = (
                    (save_every is not None)
                    and
                    (batch_interval or last or (epoch % save_every == 0))
                )
                # update changes the result to store, which is done when going to save,
                # and also when defining the metrics
                do_update = do_save or last or (metric_every is None) or (epoch % metric_every == 0)
                # change metric values when updating, as long as metric_every is set
                do_metric = do_update and (metric_every is not None)

                if do_print:
                    info = TrainEpochInfo(
                        epochs=epochs,
                        epoch=epoch,
                        start_epoch=start_epoch,
                        start=start,
                        loss=print_loss,
                        val_loss=print_val_loss,
                        accuracy=print_accuracy,
                        val_accuracy=print_val_accuracy,
                        metrics=print_metrics,
                        val_metrics=print_val_metrics,
                        count=print_count,
                        validate=bool(val_result),
                        batch_interval=batch_interval)
                    get_epoch_info = (
                        params.get_epoch_info
                        if params.get_epoch_info else
                        default_epoch_info)
                    print_str = get_epoch_info(info)

                    if print_str:
                        print(print_str)

                    print_count = 0
                    print_loss = 0
                    print_accuracy = 0
                    print_val_loss = 0
                    print_val_accuracy = 0
                    print_metrics = None

                if do_update:
                    update_results()

                    if not do_metric:
                        results.total_train_time += metric_train_time
                        results.total_val_time += metric_val_time
                        metric_train_time = 0
                        metric_val_time = 0
                    else:
                        results.total_train_time += metric_train_time
                        results.total_val_time += metric_val_time

                        results.accuracies.append(
                            (epoch, metric_accuracy / metric_count))
                        results.losses.append((epoch, metric_loss / metric_count))
                        results.times.append((epoch, metric_train_time))

                        if results.metrics is not None:
                            results.metrics.append((epoch, metrics))

                        if val_result:
                            if results.val_accuracies is not None:
                                results.val_accuracies.append(
                                    (epoch, metric_val_accuracy / metric_count))

                            if results.val_losses is not None:
                                results.val_losses.append(
                                    (epoch, metric_val_loss / metric_count))

                            if results.val_times is not None:
                                results.val_times.append((epoch, metric_val_time))

                            if results.val_metrics is not None:
                                results.val_metrics.append((epoch, val_metrics))

                        # sort by accuracy, get the last one (best case)
                        accuracies_to_use = (
                            results.val_accuracies
                            if val_result and results.val_accuracies
                            else results.accuracies)
                        best_case = sorted(accuracies_to_use, key=lambda x: x[1])[-1]
                        best_epoch, best_accuracy = best_case

                        if best_accuracy > results.best_accuracy:
                            results.best_epoch = best_epoch
                            results.best_accuracy = best_accuracy

                        metric_count = 0
                        metric_loss = 0
                        metric_accuracy = 0
                        metric_train_time = 0
                        metric_val_loss = 0
                        metric_val_accuracy = 0
                        metric_val_time = 0
                        metrics = None
                        val_metrics = None

                    if do_save:
                        full_state = self.state_handler.save_train_state(
                            params, results, state_dict)

                        if full_state:
                            save_full_state(
                                full_state,
                                save_path=params.save_path)
        except AbortedException as e:
            r = verify_early_stop(force_save=False)
            if r:
                return r
            raise e

        return results

    def _test(self, params: ATE) -> TestResult:
        test_state, state_dict = self.state_handler.load_test_state(params)
        train_results = test_state.train_results if test_state else None
        epoch = (train_results.epoch or 0) if train_results else 0
        test_results = test_state.test_results if test_state else None
        test_epoch = test_results.epoch if test_results else None

        early_stopper = params.early_stopper
        metrics_handler = self.metrics_handler

        get_batch_info = params.get_batch_info if params.get_batch_info else default_batch_info

        if not self.test_inner:
            raise Exception('test_inner is not defined')

        if not self.can_run(early_stopper):
            raise AbortedException()

        if (
            (not test_results)
            or
            test_results.batch
            or
            (test_epoch is None)
            or
            (test_epoch < epoch)
        ):
            print_str = 'Starting test...'
            separator = '=' * len(print_str)
            print(separator)
            print(print_str)

            test_results = TestResult(
                epoch=epoch,
                loss=0.0,
                accuracy=0.0,
                total_time=0,
                batch=0,
                total_batch=None,
                last_epoch_losses=None,
                last_epoch_accuracies=None,
                last_epoch_times=None)

            batch_handler = TestBatchHandler(
                params=params,
                results=test_results,
                save_state=lambda result: self.state_handler.save_test_state(
                    params, result, state_dict),
                early_stopper=early_stopper,
                metrics_handler=metrics_handler,
                get_batch_info=get_batch_info)

            test_params = ActionWrapperActionParams(
                main_params=params,
                full_state=test_state,
                epoch=epoch,
                batch_handler=batch_handler)

            with torch.set_grad_enabled(False):
                result = self.test_inner(test_params)

            if result:
                loss = result.total_loss
                accuracy = result.total_accuracy
                total_time = result.total_time

                test_results.epoch = epoch
                test_results.batch = 0
                test_results.loss = loss
                test_results.accuracy = accuracy
                test_results.total_time = total_time

                print_str = f'Test completed in {total_time/1000:.3f} seconds.'
                separator = '=' * len(print_str)
                print(separator)
                print(print_str)
                print(separator)
        else:
            if params.print_every is not None:
                print_str = f'(test epoch: {test_epoch}, last trained epoch: {epoch})'
                print_str = f'Test already completed ({print_str}).'
                print(print_str)

        if not test_results:
            raise Exception('the test returned no result')

        if test_state and test_results:
            self.state_handler.save_test_state(
                params, test_results, state_dict)

        return test_results

####################################################
################ Private Functions #################
####################################################

def _load_state_dict(save_path: str | None) -> dict[str, typing.Any] | None:
    if save_path:
        if os.path.isfile(save_path):
            checkpoint = torch.load(save_path, weights_only=True)
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

def _as_minutes(s: float, spaces: int | None = None):
    m = math.floor(s / 60)
    s -= m * 60
    info = f'{m:.0f}m {s:02.2f}s'
    return f'{info:>{spaces}}' if spaces else info

def _time_since(since: float, percent: float, spaces: int | None = None):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    passed = _as_minutes(s, spaces=spaces)
    remaining = _as_minutes(rs, spaces=spaces)
    return f'{passed} (eta: {remaining})'

####################################################
################# Public Functions #################
####################################################

def time_diff_millis(start_time: float, end_time: float) -> int:
    return int((end_time - start_time) * 1000)

def load_full_state(
        from_state_dict: typing.Callable[[dict[str, typing.Any]], S | None],
        save_path: str | None) -> S | None:
    checkpoint = _load_state_dict(save_path)

    if not checkpoint:
        return None

    state = from_state_dict(checkpoint)
    return state

def save_full_state(state: S, save_path: str | None) -> S | None:
    state_dict = _save_state_dict(
        state_dict=state.state_dict(), save_path=save_path)

    if not state_dict:
        return None

    return state
