import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

import torchmetrics
from torchmetrics import Metric
from torch.utils.data import DataLoader
from FRAIL.data.dataset import CustomDataset
from FRAIL.models.ghostnet import GhostNet


class Trainer(pl.LightningModule):
    def __init__(self, input_size, learning_rate, num_classes):
        super().__init__()
        self.input_size = input_size #probably unecessary
        self.lr = learning_rate
        self.ghostnet = GhostNet(num_classes=num_classes)
        self.loss_function = nn.CrossEntropyLoss

    def forward(self, x):
        return self.ghostnet(x)

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log(
            {
                "train_loss" : loss,
                "train_accuracy" : accuracy,
                "train_f1_score" : f1_score,
                },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss" : loss, "scores": scores, "y" : y}

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


