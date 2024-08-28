# ruff: noqa: E741 (ambiguous variable name)
import os
import typing
from typing import Callable
import torch
from torch import nn
from torch.utils.data import DataLoader
from auto_mind.supervised._dataset import DatasetGroup
from auto_mind.supervised._action_data import (
    Scheduler, EarlyStopper, EvalParams, TrainEpochInfo,
    TrainBatchInfo, TrainResult, TestResult, BatchInOutParams, GeneralHookParams,
    TrainParams, TestParams)
from auto_mind.supervised._action_handlers import (
    BatchAccuracyCalculator, BatchExecutor, Evaluator, EvaluatorParams,
    MetricsCalculatorInputParams, MetricsCalculator)
from auto_mind.supervised._batch_handlers import MetricsHandler
from auto_mind.supervised._action import GeneralAction

I = typing.TypeVar("I", bound=typing.Sized)
O = typing.TypeVar("O")
T = typing.TypeVar("T")
TG = typing.TypeVar("TG", bound=typing.Sized)
MT = typing.TypeVar("MT")
EI = typing.TypeVar("EI")
EO = typing.TypeVar("EO")
DI = typing.TypeVar("DI")
DO = typing.TypeVar("DO")

####################################################
#################### Parameters ####################
####################################################

class ManagerDataParams(typing.Generic[I, TG]):
    def __init__(
        self,
        train_dataloader: DataLoader[tuple[I, TG]],
        validation_dataloader: DataLoader[tuple[I, TG]] | None,
        test_dataloader: DataLoader[tuple[I, TG]] | None,
    ):
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader

    @classmethod
    def from_datasets(
        cls,
        datasets: DatasetGroup[typing.Any],
        batch_size: int = 64,
        limit: int | None = None,
    ) -> typing.Self:
        if limit:
            datasets = datasets.limit(limit)

        train_dataloader = DataLoader(
            datasets.train,
            batch_size=batch_size)

        validation_dataloader = DataLoader(
            datasets.validation,
            batch_size=batch_size
        ) if datasets.validation is not None else None

        test_dataloader = DataLoader(
            datasets.test,
            batch_size=batch_size
        ) if datasets.test is not None else None

        return cls(
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            test_dataloader=test_dataloader)

class ManagerModelParams(typing.Generic[I, O, TG]):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module | typing.Callable[[BatchInOutParams[I, O, TG]], torch.Tensor],
        executor: BatchExecutor[I, O],
        use_best: bool = False,
        clip_grad_max: float | None = None,
    ):
        self.model = model
        self.criterion = criterion
        self.executor = executor
        self.use_best = use_best
        self.clip_grad_max = clip_grad_max

class ManagerMetricsParams(typing.Generic[I, O, TG, MT, EI, EO]):
    def __init__(
        self,
        evaluator: Evaluator[EI, EO] | None = None,
        accuracy_calculator: BatchAccuracyCalculator[I, O, TG] | None = None,
        metrics_calculator: MetricsCalculator | None = None,
        batch_interval: bool = False,
        default_interval: int | None = 1,
        save_every: int | None = None,
        print_every: int | None = None,
        metric_every: int | None = None,
        get_epoch_info: typing.Callable[[TrainEpochInfo[MT]], str] | None = None,
        get_batch_info: typing.Callable[[TrainBatchInfo[MT]], str] | None = None,
        train_metrics_handler: MetricsHandler[I, O, TG, MT] | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.accuracy_calculator = accuracy_calculator
        self.metrics_calculator = metrics_calculator
        self.batch_interval = batch_interval
        self.default_interval = default_interval
        self.save_every = save_every
        self.print_every = print_every
        self.metric_every = metric_every
        self.get_epoch_info = get_epoch_info
        self.get_batch_info = get_batch_info
        self.train_metrics_handler = train_metrics_handler

class ManagerOptimizerParams:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Scheduler | None = None,
        step_only_on_accuracy_loss: bool = False,
        train_early_stopper: EarlyStopper | None = None,
        test_early_stopper: EarlyStopper | None = None,
    ):
        self.train_early_stopper = train_early_stopper
        self.test_early_stopper = test_early_stopper
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.step_only_on_accuracy_loss = step_only_on_accuracy_loss

class ManagerConfig(typing.Generic[I, O, TG]):
    def __init__(
        self,
        save_path: str | None = None,
        random_seed: int | None = None,
        device: torch.device | None = None,
        train_hook: Callable[[GeneralHookParams[I, O, TG]], None] | None = None,
        validation_hook: Callable[[GeneralHookParams[I, O, TG]], None] | None = None,
        test_hook: Callable[[GeneralHookParams[I, O, TG]], None] | None = None,
    ):
        self.save_path = save_path
        self.random_seed = random_seed
        self.device = device
        self.train_hook = train_hook
        self.validation_hook = validation_hook
        self.test_hook = test_hook

class ManagerTrainResult:
    def __init__(
        self,
        completed: bool,
        train_results: TrainResult | None,
        test_results: TestResult | None,
        metrics: dict[str, typing.Any] | None,
    ):
        self.train_results = train_results
        self.test_results = test_results
        self.metrics = metrics
        self.completed = completed

####################################################
##################### Manager ######################
####################################################

class Manager(typing.Generic[I, O, TG, MT, EI, EO]):
    def __init__(
        self,
        data_params: ManagerDataParams[I, TG],
        model_params: ManagerModelParams[I, O, TG],
        optimizer_params: ManagerOptimizerParams,
        metrics_params: ManagerMetricsParams[I, O, TG, MT, EI, EO],
        config: ManagerConfig[I, O, TG],
    ):
        train_dataloader = data_params.train_dataloader
        validation_dataloader = data_params.validation_dataloader
        test_dataloader = data_params.test_dataloader

        model = model_params.model
        criterion = model_params.criterion
        executor = model_params.executor
        use_best = model_params.use_best
        clip_grad_max = model_params.clip_grad_max

        optimizer = optimizer_params.optimizer
        scheduler = optimizer_params.scheduler
        step_only_on_accuracy_loss = optimizer_params.step_only_on_accuracy_loss
        train_early_stopper = optimizer_params.train_early_stopper
        test_early_stopper = optimizer_params.test_early_stopper

        evaluator = metrics_params.evaluator
        accuracy_calculator = metrics_params.accuracy_calculator
        metrics_calculator = metrics_params.metrics_calculator
        batch_interval = metrics_params.batch_interval
        default_interval = metrics_params.default_interval
        save_every = metrics_params.save_every
        print_every = metrics_params.print_every
        metric_every = metrics_params.metric_every
        get_epoch_info = metrics_params.get_epoch_info
        get_batch_info = metrics_params.get_batch_info
        train_metrics_handler = metrics_params.train_metrics_handler

        save_path = config.save_path
        random_seed = config.random_seed
        device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_hook = config.train_hook
        validation_hook = config.validation_hook
        test_hook = config.test_hook

        save_every = save_every if save_every is not None else default_interval
        print_every = print_every if print_every is not None else default_interval
        metric_every = metric_every if metric_every is not None else default_interval

        model.to(device)

        action: GeneralAction[I, O, TG, MT] = GeneralAction(
            random_seed=random_seed,
            use_best=use_best,
            executor=executor,
            accuracy_calculator=accuracy_calculator,
            metrics_calculator=metrics_calculator,
            metrics_handler=train_metrics_handler)

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader

        self.model = model
        self.criterion = criterion
        self.clip_grad_max = clip_grad_max

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.step_only_on_accuracy_loss = step_only_on_accuracy_loss
        self.train_early_stopper = train_early_stopper
        self.test_early_stopper = test_early_stopper

        self.evaluator = evaluator
        self.executor = executor
        self.batch_interval = batch_interval
        self.save_every = save_every
        self.print_every = print_every
        self.metric_every = metric_every
        self.get_epoch_info = get_epoch_info
        self.get_batch_info = get_batch_info

        self.save_path = save_path
        self.train_hook = train_hook
        self.validation_hook = validation_hook
        self.test_hook = test_hook

        self.action = action

    def clear(self) -> None:
        save_path = self.save_path
        if save_path and os.path.exists(save_path):
            os.remove(save_path)

    def info(self) -> ManagerTrainResult | None:
        save_path = self.save_path
        action_info = self.action.info(save_path) if save_path else None
        result = ManagerTrainResult(
            completed=action_info.completed,
            train_results=action_info.train_results,
            test_results=action_info.test_results,
            metrics=action_info.metrics,
        ) if action_info else None
        return result

    def train(self, epochs: int) -> ManagerTrainResult:
        if self.save_path is not None:
            self.action.define_as_pending(save_path=self.save_path)

        train_results = self._train(epochs)
        finished = (
            train_results is not None
            and (
                train_results.early_stopped
                or ((train_results.epoch or 0) >= epochs)
            )
        )
        test_results: TestResult | None = None
        metrics: dict[str, typing.Any] | None = None
        completed=False

        if train_results and finished:
            should_test = self.test_dataloader is not None

            if should_test:
                test_results = self._test()

            info = self.info()

            finished_test = test_results.epoch >= train_results.epoch if test_results else False
            completed = info.completed if info else False

            if (finished_test or not should_test) and ((not info) or (not completed)):
                metrics_params = MetricsCalculatorInputParams(
                    model=self.model,
                    save_path=self.save_path,
                )
                metrics = self.action.calculate_metrics(metrics_params)
                if self.save_path is not None:
                    self.action.define_as_completed(save_path=self.save_path)
                completed = True

        return ManagerTrainResult(
            completed=completed,
            train_results=train_results,
            test_results=test_results,
            metrics=metrics,
        )

    def _train(self, epochs: int) -> TrainResult | None:
        train_dataloader = self.train_dataloader
        validation_dataloader = self.validation_dataloader

        model = self.model
        criterion = self.criterion
        clip_grad_max = self.clip_grad_max

        optimizer = self.optimizer
        scheduler = self.scheduler
        step_only_on_accuracy_loss = self.step_only_on_accuracy_loss

        batch_interval = self.batch_interval
        save_every = self.save_every
        print_every = self.print_every
        metric_every = self.metric_every
        get_epoch_info = self.get_epoch_info
        get_batch_info = self.get_batch_info

        save_path = self.save_path
        train_hook = self.train_hook
        validation_hook = self.validation_hook

        action = self.action

        train_params = TrainParams(
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopper=self.train_early_stopper,
            step_only_on_accuracy_loss=step_only_on_accuracy_loss,
            clip_grad_max=clip_grad_max,
            epochs=epochs,
            batch_interval=batch_interval,
            save_every=save_every,
            print_every=print_every,
            metric_every=metric_every,
            get_epoch_info=get_epoch_info,
            get_batch_info=get_batch_info,
            save_path=save_path,
            train_hook=train_hook,
            validation_hook=validation_hook,
            skip_load_state=not bool(save_path),
        )

        return action.train(train_params)

    def _test(self) -> TestResult | None:
        test_dataloader = self.test_dataloader

        model = self.model
        criterion = self.criterion

        batch_interval = self.batch_interval
        save_every = self.save_every if batch_interval else None
        print_every = self.print_every if batch_interval else None
        metric_every = self.metric_every if batch_interval else None
        get_batch_info = self.get_batch_info

        save_path = self.save_path
        test_hook = self.test_hook

        action = self.action

        if test_dataloader is None:
            raise Exception('The test test dataloader is not defined')

        test_params = TestParams(
            dataloader=test_dataloader,
            model=model,
            criterion=criterion,
            early_stopper=self.test_early_stopper,
            save_every=save_every,
            print_every=print_every,
            metric_every=metric_every,
            get_batch_info=get_batch_info,
            save_path=save_path,
            hook=test_hook,
            skip_load_state=not bool(save_path),
        )

        return action.test(test_params)

    def evaluate(self, input: EI) -> EO:
        evaluator = self.evaluator
        assert evaluator is not None
        return self.debug(input, evaluator)

    def debug(self, input: DI, evaluator: Evaluator[DI, DO]) -> DO:
        with torch.no_grad():
            model = self.load_model()
            params = EvaluatorParams(model=model, input=input)
            return evaluator.run(params)

    def load_model(self) -> nn.Module:
        model = self.model
        if self.save_path:
            params = EvalParams(
                model=model,
                save_path=self.save_path)
            self.action.load_eval_state(params)
        return model

class DefaultManager(
    Manager[torch.Tensor, torch.Tensor, torch.Tensor, None, EI, EO],
    typing.Generic[EI, EO],
):
    def __init__(
        self,
        data_params: ManagerDataParams[torch.Tensor, torch.Tensor],
        model_params: ManagerModelParams[torch.Tensor, torch.Tensor, torch.Tensor],
        optimizer_params: ManagerOptimizerParams,
        metrics_params: ManagerMetricsParams[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            None,
            EI,
            EO],
        config: ManagerConfig[torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        super().__init__(
            data_params=data_params,
            model_params=model_params,
            optimizer_params=optimizer_params,
            metrics_params=metrics_params,
            config=config,
        )
