import os
import logging
from numpy import full
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.pooling import MaxPool2d
import argparse
from src.utils.common import read_yaml, create_directories

STAGE = "stage_02_base_model_creation"

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


class CNN(nn.Module):
    def __init__(self, in_, out_):
        super(CNN, self).__init__()
        logging.info("making base model........")
        self.conv_pool_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_, out_channels=8, kernel_size=5, stride=1, padding=0),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_pool_2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.Flatten = nn.Flatten()
        self.FC_01 = nn.Linear(in_features=16*4*4, out_features=128)       # <<<--- Dense layer in keras tenserflow
        self.FC_02 = nn.Linear(in_features=128, out_features=64)
        self.FC_03 = nn.Linear(in_features=64, out_features=out_)
        logging.info("base model created")

    def forward(self, x):
        logging.info("Making forwardPass....")
        x = self.conv_pool_1(x)
        x = self.conv_pool_2(x)
        x = self.Flatten(x)
        x = self.FC_01(x)
        x = F.relu(x)
        x = self.FC_02(x)
        x = F.relu(x)
        x = self.FC_03(x)
        logging.info(f"ForwardPass done ")
        return x


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", '-c', default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n************************************")
        logging.info(f">>>>>>>>>>>>>>>>>{STAGE} started<<<<<<<<<<<<<<<")
        
        config  = read_yaml(parsed_args.config)

        model_path = os.path.join(config['artifacts']['model'])
        create_directories([model_path])                                     # <<<---- create directory "artifacts"

        model_name = config['artifacts']['base_model']
        full_model_path = os.path.join(model_path, model_name)

        input = config['params']['input']
        output = config['params']['output']

        model_ob = CNN(in_=input, out_=output)
        torch.save(model_ob, full_model_path)
        logging.info(f"model created and saved at {full_model_path}")
        logging.info(f">>>>>>>>>>>>>>>>>{STAGE} completed<<<<<<<<<<<<<<<")
        logging.info("\n************************************")
        
    except Exception as e:
        logging.exception(e)
        raise e

