import torch


class Trainer:
    def __init__(
        self,
        train_epochs: int,
        lr: float = 0.02,
        optimizer: torch.optim.Optimizer = torch.optim.SGD,
    ):
        self.train_epochs = train_epochs
        self.lr = lr
        self.optimizer = optimizer

    def train(self, model, tr_data: torch.Tensor, print_epoch: int = 100):
        self.tr_data = tr_data
        self.model = model
        self.model_optimizer = self.optimizer(self.model.parameters(), lr=self.lr)
        self.model.train()
        for epoch in range(self.train_epochs):
            if epoch % print_epoch == 0:
                print("Training epoch", epoch)
            self.fit_epoch()

    def fit_epoch(self):
        # TODO Train with batches after dataloader is implemented
        loss = self.model.loss(
            self.model(self.tr_data[:, :-1]).squeeze(), self.tr_data[:, -1]
        )
        self.model_optimizer.zero_grad()
        with torch.no_grad():
            loss.backward()
            self.model_optimizer.step()
