from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing_extensions import Self

from policyflux.exceptions import EngineNotConfiguredError
from policyflux.logging_config import logger

from ..core.id_generator import get_id_generator
from ..core.layer import Layer
from ..core.types import UtilitySpace


class SequentialNeuralLayer(Layer, nn.Sequential):
    def __init__(
        self,
        input_size: int,
        output_size: int = 2,
        id: int | None = None,
        name: str = "SequentialNeuralLayer",
        architecture: list[nn.Module] | None = None,
    ) -> None:
        if id is None:
            id = get_id_generator().generate_layer_id()

        Layer.__init__(self, id, name, input_size, output_size)
        nn.Sequential.__init__(self, *(architecture or []))

        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer: torch.optim.Optimizer | None = None
        self.loss_fn: nn.Module | None = None
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.epochs: int = 1
        self.batch_size: int = 32

        # ensure model parameters live on the chosen device
        self.to(self.device)

    def call(self, bill_space: UtilitySpace, **kwargs: Any) -> float:
        tensor_input = torch.tensor(bill_space, dtype=torch.float32, device=self.device)
        output = self.forward(tensor_input)
        return float(output.squeeze().item())

    def append_neural_layer(self, layer: nn.Module) -> Self:
        super().append(layer)
        return self

    def insert_neural_layer(self, index: int, layer: nn.Module) -> Self:
        super().insert(index, layer)
        return self

    def pop_neural_layer(self, index: int = -1) -> nn.Module:
        return super().pop(index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.device:
            x = x.to(self.device)
        return super().forward(x)

    def set_nn_architecture(self, architecture: list[nn.Module]) -> None:
        self._modules.clear()
        for layer in architecture:
            self.append_neural_layer(layer)
        self.to(self.device)

    def _run_train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> float:
        if self.optimizer is None or self.loss_fn is None:
            raise EngineNotConfiguredError(
                "Optimizer and loss_fn must be set. Call compile(...) first."
            )

        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        self.train()
        self.optimizer.zero_grad()
        outputs = self.forward(inputs)

        loss = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def _run_train_loop(self) -> None:
        if self.train_loader is None:
            return
        for epoch in range(1, self.epochs + 1):
            epoch_loss: float = 0.0
            batch_count: int = 0
            for batch in self.train_loader:
                loss = self._run_train_step(batch)
                epoch_loss += loss
                batch_count += 1

            avg_loss = epoch_loss / batch_count if batch_count else 0.0
            if self.val_loader is not None:
                val_loss = self.run_validation()
                logger.info(
                    "Epoch %d/%d - train_loss=%.6f val_loss=%.6f",
                    epoch,
                    self.epochs,
                    avg_loss,
                    val_loss,
                )
            else:
                logger.info("Epoch %d/%d - train_loss=%.6f", epoch, self.epochs, avg_loss)

    def run_validation(self) -> float:
        if self.val_loader is None or self.loss_fn is None:
            return 0.0

        self.eval()
        total_loss: float = 0.0
        batch_count: int = 0
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.forward(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += float(loss.item())
                batch_count += 1

        return total_loss / batch_count if batch_count else 0.0

    def compile(
        self,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.SGD,
        lr: float = 1e-3,
        loss_fn: nn.Module | None = None,
        epochs: int = 10,
        batch_size: int = 32,
        train_dataset: torch.utils.data.Dataset | None = None,
        val_dataset: torch.utils.data.Dataset | None = None,
    ) -> None:
        # Setup training configuration
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_fn = loss_fn or nn.MSELoss()
        self.optimizer = optimizer_cls(self.parameters(), lr=lr)  # type: ignore[call-arg]

        if train_dataset is not None:
            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        if val_dataset is not None:
            self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        self.to(self.device)
