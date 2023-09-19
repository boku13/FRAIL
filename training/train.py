# imports --------------------
from model import NN
import pytorch_lightning as pl
from dataset import MnistDataModule
import config

if __name__ == '__main__':
    model = NN(learning_rate=config.learning_rate, input_size=config.input_size, num_classes=config.num_classes)
    dm = MnistDataModule(data_dir=config.data_dir, batch_size=config.batch_size, num_workers=config.num_workers)
    trainer = pl.Trainer(accelerator=config.accelerator, devices=config.devices, min_epochs=1, max_epochs=5, precision=config.precision)
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)