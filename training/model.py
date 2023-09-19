import torch    # pytorch
import torch.nn as nn    # all NN modules
import torch.optim as optim     # all optimization algorithms
import torch.nn.functional as F     # includes ReLU, tanh etc. activation functions
import pytorch_lightning as pl
import torchmetrics

class NN(pl.LightningModule):
    def __init__(self, learning_rate, input_size, num_classes):
        super().__init__()
        self.lr = learning_rate
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    # def training_epoch_end(self, outputs):
    #     pass

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters, lr=self.lr)

    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = F.cross_entropy(scores, y)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict({'train loss': loss, 'train accuracy':accuracy, 'train_f1_score': f1_score}, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        return loss
