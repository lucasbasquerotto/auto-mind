# pylint: disable=import-outside-toplevel
def test_supervised() -> None:
    import sys
    import os

    # Add the path to the package to sys.path
    package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
    if package_path not in sys.path:
        sys.path.insert(0, package_path)

    import torch
    from auto_mind import supervised # type: ignore[import-untyped]
    from auto_mind.supervised.handlers import ( # type: ignore[import-untyped]
        GeneralBatchExecutor, MaxProbBatchEvaluator, GeneralBatchAccuracyCalculator)
    from auto_mind.supervised.data import SplitData, ItemsDataset # type: ignore[import-untyped]

    # Define a simple neural network model
    class SimpleNN(torch.nn.Module):
        def __init__(self, input_size: int, hidden_size: int, num_classes: int):
            super().__init__()
            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.fc2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.relu(self.fc1(x))
            return torch.softmax(self.fc2(x), dim=1)

    input_size = 10
    hidden_size = 128
    num_classes = 3
    num_samples = 100
    epochs = 2
    seed = 1

    # Generate synthetic data
    def sample(idx: int) -> tuple[torch.Tensor, int]:
        y = idx % num_classes
        x = [float((j+1)%(y+1) == 0) for j in range(input_size)]
        return torch.tensor(x), y

    full_dataset = ItemsDataset([sample(i) for i in range(num_samples)])

    datasets = SplitData(val_percent=0.1, test_percent=0.1).split(
        full_dataset,
        shuffle=True,
        random_seed=seed)

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
    if accuracy is not None:
        min_acc = 0.999
        print(f'Test Accuracy: {accuracy * 100:.2f}%')
        assert accuracy > min_acc, f'Test Accuracy ({accuracy * 100:.2f}%) should be more than {min_acc * 100:.2f}%' # pylint: disable=line-too-long

    assert datasets.test is not None, 'Test dataset should not be None'
    X_test = torch.stack([x for x, _ in datasets.test])
    y_test = [y for _, y in datasets.test]
    eval_result = manager.evaluate(X_test).prediction
    for (_, predicted), label in zip(eval_result, y_test):
        assert predicted == label, f'Predicted: {predicted}, Label: {label}'
