# ruff: noqa: E741 (ambiguous variable name)
import time
import math
import typing
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from auto_mind.supervised._action_data import (
    BaseResult, TestResult, TrainResult, ExecutionCursor, EarlyStopper,
    TrainEarlyStopper, TrainEpochInfo, TrainBatchInfo, TrainParams,
    TestParams, Scheduler, BatchInOutParams, EvalParams, GeneralHookParams,
    StateWithMetrics)
from auto_mind.supervised._action_handlers import (
    BatchExecutor, BatchAccuracyCalculator, BatchExecutorParams, BatchAccuracyParams,
    MetricsCalculator, MetricsCalculatorInputParams, MetricsCalculatorParams,
    AbortedException)
from auto_mind.supervised._state_handlers import StateHandler
from auto_mind.supervised._batch_handlers import (
    BatchHandlerData, MetricsHandler, BatchHandler, BatchHandlerResult,
    TrainBatchHandler, TestBatchHandler, BatchHandlerRunParams)

I = typing.TypeVar("I", bound=typing.Sized)
O = typing.TypeVar("O")
T = typing.TypeVar('T')
TG = typing.TypeVar("TG", bound=typing.Sized)
M = typing.TypeVar("M", bound=nn.Module)
OT = typing.TypeVar("OT", bound=optim.Optimizer)
RV = typing.TypeVar("RV", bound=BaseResult)
MT = typing.TypeVar("MT")

S = typing.TypeVar("S", bound=BaseResult)

ATR = typing.TypeVar("ATR", bound=TrainParams[
    typing.Any, typing.Any, typing.Any, typing.Any])
ATE = typing.TypeVar("ATE", bound=TestParams[
    typing.Any, typing.Any, typing.Any, typing.Any])

AWP = typing.TypeVar("AWP")
AWS = typing.TypeVar("AWS")

####################################################
############### General Action Impl ################
####################################################

class GeneralAction(typing.Generic[I, O, TG, MT]):
    def __init__(
        self,
        random_seed: int | None,
        use_best: bool,
        executor: BatchExecutor[I, O],
        accuracy_calculator: BatchAccuracyCalculator[I, O, TG] | None,
        metrics_handler: MetricsHandler[I, O, TG, MT] | None,
        metrics_calculator: MetricsCalculator | None = None,
    ):
        main_state_handler = StateHandler[
            TrainParams[I, O, TG, MT],
            TestParams[I, O, TG, MT],
        ](use_best=use_best)

        self.state_handler = main_state_handler
        self.metrics_handler = metrics_handler
        self.metrics_calculator = metrics_calculator
        self.random_seed = random_seed
        self.executor = executor
        self.accuracy_calculator = accuracy_calculator

    def load_eval_state(self, params: EvalParams) -> None:
        self.state_handler.load_eval_state(params)

    def info(self, save_path: str) -> StateWithMetrics | None:
        return self.state_handler.info(save_path=save_path)

    def can_run(self, early_stopper: EarlyStopper | None) -> bool:
        if not early_stopper:
            return True

        return not early_stopper.check()

    def _new_train_result(self, params: ATR) -> TrainResult:
        validate = params.validation_dataloader is not None
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
            val_accuracies=[] if validate else None,
            val_losses=[] if validate else None,
            val_times=[] if validate else None,
            val_metrics=[] if validate and self.metrics_handler else None)

    def train(self, params: ATR) -> TrainResult | None:
        full_state, _ = self.state_handler.load_train_state(params)
        results: TrainResult = (
            full_state.train_results
            if full_state
            else self._new_train_result(params))

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
        train_dataloader = params.train_dataloader
        validation_dataloader = params.validation_dataloader
        validate = validation_dataloader is not None

        full_state, state_dict = self.state_handler.load_train_state(params)
        results: TrainResult = (
            full_state.train_results
            if full_state
            else self._new_train_result(params))
        start_epoch = results.epoch + 1

        print_count = 0
        print_loss: float = 0
        print_accuracy: float | None = 0
        print_val_loss: float = 0
        print_val_accuracy : float | None= 0

        # Keep track of losses and accuracies for the metrics
        metric_count = 0
        metric_loss: float = 0
        metric_accuracy: float | None = 0
        metric_train_time = 0
        metric_val_loss: float = 0
        metric_val_accuracy: float | None = 0
        metric_val_time = 0

        print_metrics: typing.Any | None = None
        print_val_metrics: typing.Any | None = None
        metrics: typing.Any | None = None

        epoch = 0
        train_loss: float = 0.0
        train_accuracy: float | None = 0.0
        val_loss: float = 0.0
        val_accuracy: float | None = 0.0
        train_metrics: typing.Any | None = None
        val_metrics: typing.Any | None = None

        get_batch_info = params.get_batch_info if params.get_batch_info else default_batch_info
        metrics_handler = self.metrics_handler

        def update_results() -> None:
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

            if not validate:
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

        def verify_early_stop(force_save: bool) -> TrainResult | None:
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

                with torch.set_grad_enabled(True):
                    result = self._run(
                        epoch=epoch,
                        is_train=True,
                        dataloader=train_dataloader,
                        model=params.model,
                        criterion=params.criterion,
                        optimizer=params.optimizer,
                        scheduler=params.scheduler,
                        clip_grad_max=params.clip_grad_max,
                        hook=params.train_hook,
                        step_only_on_accuracy_loss=params.step_only_on_accuracy_loss,
                        batch_handler=batch_handler,
                    )

                train_loss = result.total_loss
                train_accuracy = result.total_accuracy
                train_time = result.total_time
                train_metrics = result.total_metrics

                metric_loss += train_loss
                print_loss += train_loss
                if metric_accuracy is not None and train_accuracy is not None:
                    metric_accuracy += train_accuracy
                else:
                    metric_accuracy = None
                if print_accuracy is not None and train_accuracy is not None:
                    print_accuracy += train_accuracy
                else:
                    print_accuracy = None
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

                if not validate:
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

                    with torch.set_grad_enabled(False):
                        assert validation_dataloader is not None
                        val_result = self._run(
                            epoch=epoch,
                            is_train=False,
                            dataloader=validation_dataloader,
                            model=params.model,
                            criterion=params.criterion,
                            optimizer=None,
                            scheduler=params.scheduler,
                            clip_grad_max=None,
                            hook=params.validation_hook,
                            step_only_on_accuracy_loss=params.step_only_on_accuracy_loss,
                            batch_handler=batch_handler,
                        )

                    if val_result:
                        val_loss = val_result.total_loss
                        val_accuracy = val_result.total_accuracy
                        val_time = val_result.total_time
                        val_metrics_item = val_result.total_metrics

                        metric_val_loss += val_loss
                        print_val_loss += val_loss
                        if val_accuracy is not None:
                            if metric_val_accuracy is not None:
                                metric_val_accuracy += val_accuracy

                            if print_val_accuracy is not None:
                                print_val_accuracy += val_accuracy
                        else:
                            metric_val_accuracy = None
                            print_val_accuracy = None
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
                    info: TrainEpochInfo[typing.Any] = TrainEpochInfo(
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

                        if metric_accuracy is not None:
                            results.accuracies.append(
                                (epoch, metric_accuracy / metric_count))
                        results.losses.append((epoch, metric_loss / metric_count))
                        results.times.append((epoch, metric_train_time))

                        if results.metrics is not None:
                            results.metrics.append((epoch, metrics))

                        if val_result:
                            if results.val_accuracies is not None:
                                if metric_val_accuracy is not None:
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
                        best_cases = sorted(accuracies_to_use, key=lambda x: x[1])
                        best_case = best_cases[-1] if best_cases else None

                        if best_case is not None and results.best_accuracy is not None:
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
                        self.state_handler.save_train_state(
                            params,
                            results,
                            state_dict)
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
        test_results: TestResult | None = test_state.test_results if test_state else None
        test_epoch = test_results.epoch if test_results else None

        early_stopper = params.early_stopper
        metrics_handler = self.metrics_handler

        get_batch_info = params.get_batch_info if params.get_batch_info else default_batch_info

        test_dataloader = params.dataloader
        assert test_dataloader is not None, 'the test dataloader is empty'

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

            with torch.set_grad_enabled(False):
                result = self._run(
                    epoch=epoch,
                    is_train=False,
                    dataloader=test_dataloader,
                    model=params.model,
                    criterion=params.criterion,
                    optimizer=None,
                    scheduler=None,
                    clip_grad_max=None,
                    hook=params.hook,
                    step_only_on_accuracy_loss=False,
                    batch_handler=batch_handler,
                )

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
                print_str = f'test epoch: {test_epoch}, last trained epoch: {epoch}'
                print_str = f'Test already completed ({print_str}).'
                print(print_str)

        if not test_results:
            raise Exception('the test returned no result')

        if test_state and test_results:
            self.state_handler.save_test_state(
                params, test_results, state_dict)

        return test_results

    def calculate_metrics(
        self,
        params: MetricsCalculatorInputParams,
    ) -> dict[str, typing.Any] | None:
        state_handler = self.state_handler
        metrics_calculator = self.metrics_calculator
        save_path = params.save_path

        if not metrics_calculator or not save_path:
            return None

        info = state_handler.load_state_with_metrics(save_path=save_path)

        if not info:
            return None

        calc_params = MetricsCalculatorParams(info=info, model=params.model)
        metrics = metrics_calculator.run(calc_params)

        state_handler.save_metrics(metrics, save_path=save_path)

        return metrics

    def define_as_pending(self, save_path: str) -> None:
        self.state_handler.define_as_completed(completed=False, save_path=save_path)

    def define_as_completed(self, save_path: str) -> None:
        self.state_handler.define_as_completed(completed=True, save_path=save_path)

    def _run(
        self,
        epoch: int,
        is_train: bool,
        dataloader: DataLoader[tuple[I, torch.Tensor]],
        model: torch.nn.Module,
        criterion: torch.nn.Module | typing.Callable[[BatchInOutParams[I, O, TG]], torch.Tensor],
        optimizer: optim.Optimizer | None,
        scheduler: Scheduler | None,
        clip_grad_max: float | None,
        hook: typing.Callable[[GeneralHookParams[I, O, TG]], None] | None,
        step_only_on_accuracy_loss: bool,
        batch_handler: BatchHandler,
    ) -> BatchHandlerResult:
        if is_train:
            model.train()

            if not optimizer:
                raise Exception('optimizer is not defined')
        else:
            model.eval()

        result = batch_handler.run(
            dataloader=dataloader,
            fn=lambda params: self._run_batch(
                params=params,
                epoch=epoch,
                is_train=is_train,
                hook=hook,
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                clip_grad_max=clip_grad_max,
            ),
            epoch=epoch,
            random_seed=self.random_seed)

        if scheduler:
            new_accuracy = result.total_accuracy

            if new_accuracy is None:
                scheduler.step()
            else:
                best_accuracy = batch_handler.best_accuracy
                worse_accuracy = (
                    new_accuracy is not None
                    and best_accuracy is not None
                    and best_accuracy > new_accuracy)

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
        executor = self.executor
        accuracy_calculator = self.accuracy_calculator

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

def _as_minutes(s: float, spaces: int | None = None) -> str:
    m = math.floor(s / 60)
    s -= m * 60
    info = f'{m:.0f}m {s:02.2f}s'
    return f'{info:>{spaces}}' if spaces else info

def _time_since(since: float, percent: float, spaces: int | None = None) -> str:
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

def default_batch_info(info: TrainBatchInfo[typing.Any]) -> str:
    print_loss = info.loss or 0
    print_accuracy = info.accuracy
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

    if print_accuracy is not None:
        acc_str = f'{100.0 * print_accuracy / print_count:>5.1f}%'
        acc_str = f'[accuracy: {acc_str}]'
    else:
        acc_str = ''

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

def default_epoch_info(info: TrainEpochInfo[typing.Any]) -> str:
    epochs = info.epochs or 0
    epoch = info.epoch or 0
    start_epoch = info.start_epoch or 0
    start = info.start or time.time()
    loss = info.loss or 0
    val_loss = info.val_loss or 0
    accuracy = info.accuracy
    val_accuracy = info.val_accuracy
    count = info.count or 1
    validate = info.validate
    batch_interval = info.batch_interval

    train_loss_avg = loss / count
    val_loss_avg = val_loss / count
    loss_avg = (
        f'[val_loss: {val_loss_avg:.4f}, train_loss: {train_loss_avg:.4f}]'
        if validate
        else f'[loss: {train_loss_avg:.4f}]')

    train_acc_str = f'{100.0 * accuracy / count:>5.1f}%' if accuracy is not None else ''
    val_acc_str = (
        f'{100.0 * val_accuracy / count:>5.1f}%'
        if validate and (val_accuracy is not None)
        else '')

    if train_acc_str or val_acc_str:
        acc_str = (
            f'[val_accuracy: {val_acc_str}, train_accuracy: {train_acc_str}]'
            if train_acc_str and val_acc_str
            else (
                f'[accuracy: {train_acc_str}]'
                if train_acc_str
                else f'[val_accuracy: {val_acc_str}]'
            )
        )
    else:
        acc_str = ''

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
