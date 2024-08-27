# ruff: noqa: E741 (ambiguous variable name)
# pylint: disable=abstract-method
# mypy: disable-error-code=attr-defined
import typing
from typing import Any, Callable, Generator, Generic, Iterable, Sized, TypeVar
import torch
from torch.utils.data import Dataset, IterableDataset, Subset
from torch.utils.data import random_split

T = TypeVar("T")
I = TypeVar("I")
O = TypeVar("O")

class DatasetGroup(Dataset[I], Generic[I]):
    def __init__(
            self,
            train: Dataset[I],
            validation: Dataset[I] | None,
            test: Dataset[I] | None):
        self.train = train
        self.validation = validation
        self.test = test

    def transform(self, transform: Callable[[I], O]) -> 'DatasetGroup[O]':
        return DatasetGroup(
            train=DatasetTransformer(
                dataset=self.train,
                transform=transform),
            validation=DatasetTransformer(
                dataset=self.validation,
                transform=transform) if self.validation else None,
            test=DatasetTransformer(
                dataset=self.test,
                transform=transform) if self.test else None)

    def filter(self, filter: Callable[[I], bool]) -> typing.Self:
        return self.__class__(
            train=DatasetFilter(
                dataset=self.train,
                filter=filter),
            validation=DatasetFilter(
                dataset=self.validation,
                filter=filter) if self.validation else None,
            test=DatasetFilter(
                dataset=self.test,
                filter=filter) if self.test else None)

    def limit(self, limit: int | None) -> typing.Self:
        if not limit:
            return self

        train_iter: list[I] = []
        validation_iter: list[I] = []
        test_iter: list[I] = []

        idx = 0

        for item in self.train:
            if idx >= limit:
                break

            train_iter.append(item)
            idx += 1

        idx = 0

        if self.validation is not None:
            for item in self.validation:
                if idx >= limit:
                    break

                validation_iter.append(item)
                idx += 1

        idx = 0

        if self.test is not None:
            for item in self.test:
                if idx >= limit:
                    break

                test_iter.append(item)
                idx += 1

        return self.__class__(
            train=ItemsDataset(train_iter),
            validation=(
                ItemsDataset(validation_iter)
                if self.validation is not None
                else None),
            test=ItemsDataset(test_iter) if self.test is not None else None)

    def sized(self) -> typing.Self:
        train_ok = isinstance(self.train, Sized)
        validation_ok = isinstance(self.validation, Sized) if self.validation else True
        test_ok = isinstance(self.test, Sized) if self.test else True

        if train_ok and validation_ok and test_ok:
            return self

        train_list = [d for d in self.train]
        validation_list = [d for d in self.validation] if self.validation else None
        test_list = [d for d in self.test] if self.test else None

        return self.__class__(
            train=ItemsDataset(train_list),
            validation=(
                ItemsDataset(validation_list)
                if validation_list is not None
                else None),
            test=ItemsDataset(test_list) if test_list is not None else None)


class SplitData():
    def __init__(self, test_percent: float | None, val_percent: float | None):
        self.test_percent = test_percent
        self.val_percent = val_percent

    def split(
        self,
        dataset: Dataset[I],
        shuffle: bool | None = None,
        random_seed: int | None = None,
    ) -> DatasetGroup[I]:
        if not isinstance(dataset, typing.Sized):
            raise ValueError("Dataset must be sized")

        amount = len(dataset)

        test_amount = (
            int(amount * self.test_percent)
            if self.test_percent is not None else 0)
        val_amount = (
            int(amount * self.val_percent)
            if self.val_percent is not None else 0)
        train_amount = amount - test_amount - val_amount

        if shuffle:
            train_dataset, val_dataset, test_dataset = random_split(
                dataset,
                (train_amount, val_amount, test_amount),
                generator=(
                    torch.Generator().manual_seed(random_seed)
                    if random_seed
                    else None))
        else:
            train_dataset = Subset(dataset, range(train_amount))
            val_dataset = Subset(dataset, range(
                train_amount, train_amount + val_amount))
            test_dataset = Subset(dataset, range(
                train_amount + val_amount, amount))

        return DatasetGroup(
            train=train_dataset,
            validation=val_dataset,
            test=test_dataset)

class ItemsDataset(Dataset[I], Generic[I]):
    def __init__(self, items: list[I]):
        self.items = items

    def __getitem__(self, idx: int) -> I:
        item = self.items[idx]
        return item

    def __len__(self) -> int:
        return len(self.items)


class IterDataset(IterableDataset[I], Generic[I]):
    def __init__(self, generator: Callable[[], Generator[I, Any, None]]):
        self.generator = generator

    def __iter__(self) -> Generator[I, typing.Any, None]:
        return self.generator()


class DirectIterableDataset(IterableDataset[I], Generic[I]):
    def __init__(self, iterable: Iterable[I]):
        self.iterable = iterable

    def __iter__(self) -> typing.Iterator[I]:
        return iter(self.iterable)


class DatasetTransformer(IterDataset[O], Generic[I, O]):
    def __init__(self, dataset: Dataset[I], transform: Callable[[I], O]):
        def generator() -> typing.Generator[O, None, None]:
            for item in dataset:
                yield transform(item)

        super().__init__(generator=generator)

        if isinstance(dataset, Sized):
            self.__len__ = dataset.__len__

class DatasetFilter(IterDataset[I], Generic[I]):
    def __init__(
        self,
        dataset: Dataset[I],
        filter: Callable[[I], bool],
    ):
        def generator() -> typing.Generator[I, None, None]:
            for item in dataset:
                if filter(item):
                    yield item

        super().__init__(generator)
