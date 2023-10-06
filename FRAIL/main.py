import argparse
import yaml
import torch
import torch.optim as optim
import pytorch_lightning as pl

from FRAIL.models.ghostnet import ghost_net
from FRAIL.data.dataset import CustomDataset
from FRAIL.training.train import Trainer
# from FRAIL.training.logs import 

def override_config(args, dict_param):
    for k, v in dict_param.items():
        if isinstance(v, dict):
            args = override_config(args, v)
        else:
            setattr(args, k, v)
    return args

def load_yaml(path):
    with open(path, 'r') as f:
        d = yaml.safe_load(f)
    return d  

def main(args):
    
    model = ghost_net()
    data = CustomDataset(args.data_dir, batch_size = args.batch_size, num_workers = args.num_workers)
    
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, min_epochs=args.epochs)
    trainer.fit(model, data)
    trainer.validate(model, data)
    trainer.test(model, data)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--yaml', default='config/ghostnet.yaml', type=str)
    
    args = parser.parse_args()
    
    if(args.yaml is not None):
        dict_param = load_yaml(args.yaml)
        args = override_config(args, dict_param)
        
    main(args)


    
