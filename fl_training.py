from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import hydra
import os
import copy 
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from parsing_utils import make_omegaconf_resolvers
from hydra.utils import instantiate
from pathlib import Path
from uuid import uuid4

import flwr as fl

def get_parameters(net) -> List[np.ndarray]:
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)





def train(cfg, model, cid):
    # seeding
    if cfg.seed:
        seed_everything(cfg.seed)
        cfg.trainer.benchmark = False
        cfg.trainer.deterministic = True

    # setup logger
    try:
        Path(
            "./main.log"
        ).unlink()  # gets automatically created, however logs are available in Weights and Biases so we do not need to log twice
    except:
        pass
    log_path = Path(cfg.trainer.logger.save_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    cfg.trainer.logger.group = str(uuid4())

    # uid = hydra.core.hydra_config.HydraConfig.get().output_subdir.split("/")[-1]
    # cfg.trainer.logger.group = uid

    # add sync_batchnorm if multiple GPUs are used
    if cfg.trainer.devices > 1 and cfg.trainer.accelerator == "gpu":
        cfg.trainer.sync_batchnorm = True

    # remove callbacks that are not enabled
    cfg.trainer.callbacks = [i for i in cfg.trainer.callbacks.values() if i]
    if not cfg.trainer["enable_checkpointing"]:
        cfg.trainer.callbacks = [
            i
            for i in cfg.trainer.callbacks
            if i["_target_"] != "lightning.pytorch.callbacks.ModelCheckpoint"
        ]
    
    cfg.trainer.max_epochs = 1
    trainer = instantiate(cfg.trainer)

    cfg.data.module.data_root_dir = f'/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/datasets/512/FL_Clients/client_{cid}'
    dataset = instantiate(cfg.data).module

    # log hypperparams
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["model"].pop("_target_")
    cfg_dict["model"]["model"] = cfg_dict["model"].pop("name")
    trainer.logger.log_hyperparams(cfg_dict["model"])
    cfg_dict["data"]["module"].pop("_target_")
    cfg_dict["data"]["module"]["train_transforms"] = ".".join(
        cfg_dict["data"]["module"]["train_transforms"]["_target_"].split(".")[-2:]
    )
    cfg_dict["data"]["module"]["test_transforms"] = ".".join(
        cfg_dict["data"]["module"]["test_transforms"]["_target_"].split(".")[-2:]
    )
    cfg_dict["data"]["module"].pop("name")
    trainer.logger.log_hyperparams(cfg_dict["data"]["module"])

    # start fitting
    trainer.fit(model, dataset)
    wandb.finish()

def test(cfg, model, cid):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    # instantiate the dataset from the config if not some other dataset is specified in the infer.yaml
    cfg.data.module.data_root_dir = f'/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/datasets/512/FL_Clients/client_{cid}'
    
    # Instantiate the data module from the configuration
    data_module = instantiate(cfg.data.module)
    
    # Prepare the data (calling setup with 'test' stage to load validation data)
    data_module.setup('test')
    
    # Retrieve the validation dataset
    val_dataset = data_module.val_dataset
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    # Instantiate the trainer
    trainer = instantiate(cfg.trainer)
    
    # Run the prediction on the validation dataset
    predictions = trainer.predict(model, val_loader)
    
    y_true = []
    y_pred = []

    for batch in predictions:
        y_batch, y_hat_batch = batch
        y_true.append(y_batch)
        y_pred.append(y_hat_batch.argmax(dim=1))  # Assuming y_hat_batch is logits, take argmax for predictions
        loss += criterion(y_hat_batch.argmax(dim=1), y_batch).item()

    y_true = torch.cat(y_true).cpu().numpy()
    y_pred = torch.cat(y_pred).cpu().numpy()
    
    # Calculate accuracy and F1 score
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')  # Use 'weighted' to account for class imbalance
    loss /= len(val_loader)

    return loss, len(val_loader), accuracy, f1

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, cfg):
        self.cid = cid
        self.net = net
        self.cfg = cfg

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")
        set_parameters(self.net, parameters)
        train(self.cfg, self.net, self.cid)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, len_val, accuracy, f1 = test(self.cfg, self.net, self.cid)
        return float(loss), len_val, {"accuracy": float(accuracy), "f1": float(f1)}

def get_strategy(strat_name):

    strategy = fl.server.strategy.FedAvg(
                    fraction_fit=1,  # Sample 50% of available clients for training each round
                    fraction_evaluate=1,  # No federated evaluation
                    # on_fit_config_fn=fit_config,
                    # evaluate_fn=get_evaluate_fn(centralized_testset),  # Global evaluation function
                )
    return strategy

class FL_Trainer():
    def __init__(self, **kwargs):
        super(FL_Trainer, self).__init__(**kwargs)

    def run_fl(self):

        

        @hydra.main(version_base=None, config_path="./cli_configs", config_name="train")
        def main(cfg):

            DEVICE = torch.device(cfg.trainer.accelerator)  # Try "cuda" to train on GPU
            print(
                f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")

            def client_fn(cid) -> FlowerClient:

                net = instantiate(cfg.model, _recursive_=False).to(DEVICE) #ConvNext(**cfg.model) 
                return FlowerClient(cid, net, cfg)

            # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
            client_resources = None
            if DEVICE.type == "cuda":
                client_resources = {"num_gpus": 1}
            
            fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=cfg.fl.clients,
                config=fl.server.ServerConfig(num_rounds=cfg.fl.num_rounds),
                client_resources=client_resources,
                strategy= get_strategy(cfg.fl.strategy)
            )
        
        main()


if __name__ == "__main__":
    make_omegaconf_resolvers()

    fler = FL_Trainer()
    fler.run_fl()
    