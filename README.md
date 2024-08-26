# AutoMind: A Comprehensive Machine Learning Library

**AutoMind** is a flexible and extensible Python package designed to streamline the development and deployment of machine learning solutions. At its core, the package features a powerful manager classes that orchestrate various machine learning workflows, allowing users to focus on their specific applications without getting bogged down by implementation details. The package is built to be adaptable, enabling seamless integration of custom algorithms and models.

### Key Features:
- **Supervised Learning Management**: Effortlessly handle the training process for models with labeled datasets. While basic algorithms are provided, the focus is on managing and optimizing the workflow.
- **Reinforcement Learning Orchestration** *(to be done)*: A robust framework for managing RL environments and training processes, making it easy to experiment and deploy RL agents.
- **Semantic Processing Coordination** *(to be done)*: Tools for handling the end-to-end process of vectorizing meanings, processing them through neural architectures, and decoding them into useful formats.

Whether you're building traditional models, exploring reinforcement learning, or working with complex semantic vectors, **AutoMind** provides the infrastructure to manage your projects efficiently while allowing room for customization and expansion.

## Explore AutoMind Examples

To see **AutoMind** in action, explore our dedicated repository for examples and tutorials: [auto-mind-examples](https://github.com/lucasbasquerotto/auto-mind-examples).

This repository contains a variety of examples, including:

- **Supervised Learning**: Learn how to manage and train models using labeled datasets.
- **Reinforcement Learning** *(to be done)*: Set up RL environments, train agents, and analyze their performance.
- **Semantic Processing** *(to be done)*: Work with vectorized meanings and process semantic information.

Whether you're getting started or looking to expand your understanding of the **AutoMind** package, these examples will provide valuable insights and practical guidance.

## Supervised Learning - Usage

This section demonstrates how to use the provided code for supervised learning tasks. Supervised learning involves training a model on a labeled dataset, where the input data is paired with the correct output. The model learns to map inputs to outputs and can then make predictions on new, unseen data.

Below is a demo example of how to set up and train a supervised learning model using this codebase. The example uses synthetic data that reaches 100% accuracy and is intended to provide an initial introduction to the manager class. For more complex cases, please refer to the repository [auto-mind-examples](https://github.com/lucasbasquerotto/auto-mind-examples).

```python
import torch
from auto_mind import supervised
from auto_mind.supervised.handlers import GeneralBatchExecutor, MaxProbBatchEvaluator, GeneralBatchAccuracyCalculator
from auto_mind.supervised.data import SplitData, ItemsDataset

# Define a simple neural network model
class SimpleNN(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=1)

input_size = 10
hidden_size = 128
num_classes = 3
num_samples = 100
epochs = 2
seed = 1

# Generate synthetic data
def sample(idx: int):
    y = idx % num_classes
    x = [float((j+1)%(y+1) == 0) for j in range(input_size)]
    return torch.tensor(x), y

full_dataset = ItemsDataset([sample(i) for i in range(num_samples)])

datasets = SplitData(val_percent=0.1, test_percent=0.1).split(full_dataset, shuffle=True, random_seed=seed)

torch.manual_seed(seed)

# Initialize the model, loss function, and optimizer
model = SimpleNN(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)

manager = supervised.Manager(
    data_params=supervised.ManagerDataParams.from_datasets(
        datasets=datasets,
        batch_size=num_samples // 20,
    ),
    model_params=supervised.ManagerModelParams(
        model=model,
        criterion=torch.nn.CrossEntropyLoss(),
        executor=GeneralBatchExecutor(),
        use_best=False,
    ),
    optimizer_params=supervised.ManagerOptimizerParams(
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
    ),
    metrics_params=supervised.ManagerMetricsParams(
        evaluator=MaxProbBatchEvaluator(executor=GeneralBatchExecutor()),
        accuracy_calculator=GeneralBatchAccuracyCalculator(),
        batch_interval=True,
        default_interval=1,
    ),
    config=supervised.ManagerConfig(
        save_path=None,
        random_seed=seed,
    ),
)

info = manager.train(epochs=epochs)

assert info is not None, 'Info should not be None'
assert info.test_results is not None, 'Test results should not be None'

accuracy = info.test_results.accuracy
min_acc = 0.999
print(f'Test Accuracy: {accuracy * 100:.2f}%')
assert accuracy > min_acc, f'Test Accuracy ({accuracy * 100:.2f}%) should be more than {min_acc * 100:.2f}%'

assert datasets.test is not None, 'Test dataset should not be None'
X_test = torch.stack([x for x, _ in datasets.test])
y_test = [y for _, y in datasets.test]
eval_result = manager.evaluate(X_test).prediction
for (_, predicted), label in zip(eval_result, y_test):
    assert predicted == label, f'Predicted: {predicted}, Label: {label}'
```

## Supervised Learning - Manager Class

The `Manager` class is responsible for orchestrating the training, validation, and testing of machine learning models. It can also be used to evaluate an already trained model. It integrates various components such as data loaders, models, optimizers, and metrics to provide a streamlined interface for supervised learning tasks.

The `Executor` is a component used to run a batch of data through the model. It is responsible for executing the forward pass of the model with the given input data and returning the output. The `Executor` can be customized to handle different types of input and output processing, making it versatile for various tasks. The full output returned by the `run` method can be passed across batches within the same epoch, if needed, and is defined in the `last_output` property of the argument passed to this method.

For example, the [GeneralBatchExecutor](/src/auto_mind/supervised/_general_action.py#L122) is the simplest case: it implements the `run` method to pass the input data to the model and return the model's output. This output is also the final output returned by the `main_output()` method, which should be a `Tensor` and will be passed to the loss function (that by default expects an output and a target, both tensors).

The executor of the [NLP From Scratch: Classifying Names with a Character-Level RNN](https://github.com/lucasbasquerotto/auto-mind-examples/blob/main/supervised/char_rnn_classification.ipynb) example returns the output along with a `hidden` tensor in the `run` method (the full output is a tuple of two tensor elements). However, the `main_output` method only returns the first element of the tuple (the actual output) to be used when calculating the loss. In this specific case, tough, the output could be returned alone in the `run` method because the `hidden` tensor is not reused across batches. However, defining it this way is useful, for example, when evaluating the model and wanting to retrieve the hidden tensor (for instance, to inspect it).

The manager expects a single model. If more than one model is trained for a task, they should be included as components of a larger model. The manager also expects a single optimizer, and an [OptimizerChain](/src/auto_mind/supervised/_action.py#L99) can be used to allow several optimizers.

The example [NLP From Scratch: Translation with a Sequence to Sequence Network and Attention](https://github.com/lucasbasquerotto/auto-mind-examples/blob/main/supervised/seq2seq_translation.ipynb), from the examples repository, has the encoder and decoder used together in the `EncoderDecoder` model class, and an optimizer chain with both the encoder and decoder optimizers.

The criterion passed to the manager is the loss function and can either be a torch module that receives the output (returned by the `main_output()` method of the executor) and the target batch from the dataloader, or a function that expects a single parameter of type [BatchInOutParams](/src/auto_mind/supervised/_action.py#L667) that has properties related to the input, target, and the outputs of the executor, and should return a tensor. The function form is useful if you want to use more data generated in the executor, not just a single output tensor, to calculate the loss (as defined in the example [TorchVision Object Detection Finetuning Tutorial](https://github.com/lucasbasquerotto/auto-mind-examples/blob/main/supervised/torchvision_tutorial.ipynb)).

The `Evaluator` is an important component used to leverage the model for various tasks. For instance, if you want `manager.evaluate()` to return the string of a category based on an input that is the URL of an image, you can create an evaluator that performs the following steps: loads the image from the URL, converts it into a tensor, passes the tensor to the model, retrieves the category index from the model's output tensor, and then maps the index to the category name to return it. The `Evaluator` is versatile and not necessarily focused on performance assessment; it can handle a wide range of tasks involving model usage, including preprocessing inputs and post-processing outputs. It can even be used in scenarios where the goal is to just print or plot results rather than return a value. That said, the `Evaluator` is an optional component, and any actions can be performed on the trained model by calling the `load_model()` method first and using the model.

Both training and testing can use an `EarlyStopper` to determine if the process should be stopped. There are two types of early stoppers:

1. **General EarlyStopper**: This can be used in both training and testing. It should implement the `check(self) -> bool` method of the `EarlyStopper` base class to verify if the process should be temporarily stopped. This temporary stop will cease the entire process and return a partial result. The training can be continued later. This is useful for stopping a long-running process to release some resources and continue in the future, for example.

2. **TrainEarlyStopper**: This is specific to training and behaves similarly to the general early stopper but with additional functionality. It should implement the `check_finish(self) -> bool` method to stop the process completely, considering that it has ended. It should also implement `update_epoch(self, loss, accuracy, metrics)` to update the state with metrics related to the epoch (to be able to determine if it should stop). Additionally, it should implement `state_dict(self) -> dict` and `load_state_dict(self, state_dict) -> typing.Self` to manage the state of the early stopper and allow the persistence of the state even if the run is interrupted. The loss and accuracy considered are the validation metrics if the training includes a validation step; otherwise, the training metrics are used.

One example of a `TrainEarlyStopper` is the [AccuracyEarlyStopper](/src/auto_mind/supervised/_action.py#L75), which stops and finishes the training after the accuracy meets or exceeds a minimum threshold consecutively across a specified number of epochs. You can use more than one early stopper (either a basic `EarlyStopper`, a `TrainEarlyStopper`, or a combination of both) by passing them to the constructor of [ChainedEarlyStopper](/src/auto_mind/supervised/_action.py#L47).

### Manager Class

| Attribute | Description |
|-----------|-------------|
| `data_params` | Parameters related to data loading and splitting. |
| `model_params` | Parameters related to the model, loss function, and execution. |
| `optimizer_params` | Parameters related to the optimizer and learning rate scheduler. |
| `metrics_params` | Parameters related to metrics calculation and evaluation. |
| `config` | Configuration parameters such as save paths and device settings. |

### Constructor Parameters

#### ManagerDataParams

| Parameter | Type | Description |
|-----------|------|-------------|
| `train_dataloader` | `DataLoader` | DataLoader for the training dataset. |
| `validation_dataloader` | `DataLoader` \| `None` | DataLoader for the validation dataset. |
| `test_dataloader` | `DataLoader` \| `None` | DataLoader for the test dataset. |

#### ManagerModelParams

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `nn.Module` | The neural network model to be trained. |
| `criterion` | `nn.Module` | The loss function. |
| `executor` | `BatchExecutor` | Executor for batch operations. See [GeneralBatchExecutor](/src/auto_mind/supervised/_general_action.py#L122) for the simplest case.  |
| `use_best` | `bool` | Whether to use the best model based on validation performance when evaluating. Defaults to `False`. |
| `clip_grad_max` | `float` \| `None` | Maximum gradient clipping value. |

#### ManagerOptimizerParams

| Parameter | Type | Description |
|-----------|------|-------------|
| `optimizer` | `torch.optim.Optimizer` | The optimizer for training the model. |
| `scheduler` | `Scheduler` \| `None` | Learning rate scheduler. |
| `step_only_on_accuracy_loss` | `bool` | Whether to step the scheduler only on accuracy loss. |
| `train_early_stopper` | `EarlyStopper` \| `None` | Early stopper for training. |
| `test_early_stopper` | `EarlyStopper` \| `None` | Early stopper for testing. |

#### ManagerMetricsParams

| Parameter | Type | Description |
|-----------|------|-------------|
| `evaluator` | `Evaluator` \| `None` | Evaluator for utilizing the model in various tasks, including input preprocessing and output post-processing. |
| `accuracy_calculator` | `BatchAccuracyCalculator` \| `None` | Calculator for batch accuracy (defaults to [GeneralBatchAccuracyCalculator](/src/auto_mind/supervised/_general_action.py#L147)). |
| `metrics_calculator` | `MetricsCalculator` \| `None` | Calculator for additional metrics. |
| `batch_interval` | `bool` | Whether to calculate metrics and execute other actions (like saving) at batch intervals (instead of epochs). Defaults to `False`. |
| `default_interval` | `int` \| `None` | Default interval for metric calculation and other actions (like saving). |
| `save_every` | `int` \| `None` | Interval for saving the model (defaults to `default_interval`). |
| `print_every` | `int` \| `None` | Interval for printing training information (defaults to `default_interval`). |
| `metric_every` | `int` \| `None` | Interval for calculating metrics (defaults to `default_interval`). |
| `get_epoch_info` | `Callable` \| `None` | Custom function to retrieve epoch information (to be printed), overriding the default behavior. |
| `get_batch_info` | `Callable` \| `None` | Custom function to retrieve batch information (to be printed), overriding the default behavior. |
| `train_metrics_handler` | `MetricsHandler` \| `None` | Handler for training metrics, updated periodically (according to `metric_every`). |

#### ManagerConfig

| Parameter | Type | Description |
|-----------|------|-------------|
| `save_path` | `str` \| `None` | Path to save the model (`.pth` file). |
| `random_seed` | `int` \| `None` | Random seed for reproducibility (it may be necessary to call `torch.manual_seed(seed)` and similar functions before creating the model and performing any stochastic operations outside the manager to ensure full reproducibility). |
| `device` | `torch.device` \| `None` | Device to run the model on (CPU, GPU). |
| `train_hook` | `Callable` \| `None` | Hook function for training. |
| `validation_hook` | `Callable` \| `None` | Hook function for validation. |
| `test_hook` | `Callable` \| `None` | Hook function for testing. |

### Public Methods

| Method | Description |
|--------|-------------|
| `clear()` | Deletes the file that contains the model weights and training information. |
| `info()` | Returns an object with information about the training, tests, and metrics. |
| `train(epochs: int)` | Trains the model for a specified number of epochs. |
| `evaluate(input: EI) -> EO` | Evaluates the model on the given input, according to the `Evaluator` passed to the manager. |
| `debug(input: DI, evaluator: Evaluator[DI, DO]) -> DO` | Debugs the model using the provided evaluator. |
| `load_model()` | Loads the model from the specified save path. |