import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else 'cpu'


def accuracy_fn(y_true, y_predicted):
    correct = torch.eq(y_true, y_predicted).sum().item()
    return (correct / len(y_predicted)) * 100


class MultiClassClassificationModel(nn.Module):
    def __init__(self, input_features: int, output_features: int, hidden_units=8, learning_rate=0.1):
        """
        Initializes multiclass classification model

        :param input_features (int): number of features that goes into the neural network
        :param output_features (int): number of output classes
        :param hidden_units: number of neurons that each hidden layers will have, default 8
        :param learning_rate: learning rate of the gradient descent
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(params=self.parameters(), lr=learning_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def training_phase(self, x_train, y_train):
        self.train()

        y_logits = self(x_train)
        y_predictions = torch.softmax(y_logits, dim=1).argmax(dim=1)

        loss = self.loss_fn(y_logits, y_train)
        accuracy = accuracy_fn(y_true=y_train, y_predicted=y_predictions)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, accuracy

    def testing_phase(self, x_test, y_test):
        self.eval()
        with torch.inference_mode():
            test_logits = self(x_test)
            test_predictions = torch.softmax(test_logits, dim=1).argmax(dim=1)

            test_loss = self.loss_fn(test_logits, y_test)
            test_accuracy = accuracy_fn(y_true=y_test, y_predicted=test_predictions)

        return test_loss, test_accuracy

    def train_model(self, x_train: torch.Tensor, x_test: torch.Tensor, y_train: torch.Tensor, y_test: torch.Tensor,
                    epochs=1000):
        """
        Trains the model by using PyTorch's training loop pattern

        :param x_train: train dataset (tensor)
        :param x_test: test dataset (tensor)
        :param y_train: train dataset labels (tensor)
        :param y_test: test dataset labels (tensor)
        :param epochs: number of epochs
        """

        torch.manual_seed(42)
        x_train, x_test = x_train.to(device), x_test.to(device)
        y_train, y_test = y_train.to(device), y_test.to(device)

        for epoch in range(epochs):
            # Training phase
            loss, accuracy = self.training_phase(x_train, y_train)

            # Testing phase
            test_loss, test_accuracy = self.testing_phase(x_test, y_test)

            # Print out training results
            if epoch % 100 == 0:
                print(
                    f"Epoch: {epoch} | Loss: {loss:.4f} | Acc: {accuracy:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.2f}%")
