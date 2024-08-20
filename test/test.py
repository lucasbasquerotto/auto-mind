import sys
import os

# Add the path to the package to sys.path
package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
if package_path not in sys.path:
    sys.path.insert(0, package_path)

import torch
from synth_mind import supervised
from synth_mind.supervised.handlers import GeneralBatchExecutor, MaxProbBatchEvaluator, GeneralBatchAccuracyCalculator
from synth_mind.supervised.data import SplitData, ItemsDataset

# Define a simple neural network model
class SimpleNN(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=1)

# Generate synthetic data
input_size = 10
hidden_size = 32
num_classes = 3
num_samples = 100

def sample(idx: int):
    y = idx % num_classes
    x = [float((y+1)*(j+1)) for j in range(input_size)]
    return torch.tensor(x), y

full_dataset = ItemsDataset([sample(i) for i in range(num_samples)])

datasets = SplitData(val_percent=0.1, test_percent=0.1).split(full_dataset, shuffle=True, random_seed=0)

# Initialize the model, loss function, and optimizer
model = SimpleNN(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)

manager = supervised.Manager(
    data_params=supervised.Manager.data_from_datasets(
        datasets=datasets,
        batch_size=num_samples // 5,
    ),
    model_params=supervised.ManagerModelParams(
        model=model,
        criterion=torch.nn.CrossEntropyLoss(),
        executor=GeneralBatchExecutor(),
        use_best=False,
    ),
    optimizer_params=supervised.ManagerOptimizerParams(
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        scheduler=None,
        train_early_stopper=None,
        test_early_stopper=None,
    ),
    metrics_params=supervised.ManagerMetricsParams(
        evaluator=MaxProbBatchEvaluator(executor=GeneralBatchExecutor()),
        accuracy_calculator=GeneralBatchAccuracyCalculator(),
        batch_interval=False,
        default_interval=500,
    ),
    config=supervised.ManagerConfig(
        save_path=None,
        random_seed=0,
        train_hook=None,
    ),
)

info = manager.train(epochs=10000)

assert info is not None
assert info.test_results is not None

print(f'Test Accuracy: {info.test_results.accuracy * 100:.2f}%')
assert info.test_results.accuracy > 0.999

assert datasets.test is not None
X_test = torch.stack([x for x, _ in datasets.test])
y_test = [y for _, y in datasets.test]
eval_result = manager.evaluate(X_test).prediction
for (_, predicted), label in zip(eval_result, y_test):
    assert predicted == label
