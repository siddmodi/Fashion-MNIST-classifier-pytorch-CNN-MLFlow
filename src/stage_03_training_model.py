from calendar import EPOCH
from cgi import test
import logging
import os
from flask import Config
import torch
from tqdm import tqdm               
import torch.nn as nn
# import torch.nn as nn
import argparse
from src import stage_01_get_data
from src.stage_02_base_model_creation import CNN
from src.utils.common import read_yaml, create_directories


STAGE = "stage_03_training_model"


logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    config = read_yaml(config_path)
    try:
        DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
        base_model_path = os.path.join(config['artifacts']['model'], config['artifacts']['base_model'])
        logging.info(f"loading model {base_model_path}")

        loaded_model = torch.load(base_model_path)
        loaded_model.eval()
        logging.info(f'{base_model_path} model loaded')

        # load model in cuda or cpu
        loaded_model.to(DEVICE)
        logging.info(f"{loaded_model} is loaded in {DEVICE}")

        # Credentials
        LR = config['params']['LR']
        optimizer = torch.optim.Adam(loaded_model.parameters(), lr=LR)                      # <<---- optimizer
        train_data_loader, test_data_loader, label_map = stage_01_get_data.main(config_path)
        EPOCHS = config['Epoch']
        criterion = nn.CrossEntropyLoss()                                                   # <<---- loss function

        for epoch in range(EPOCHS):
            with tqdm(train_data_loader) as tqdm_epoch:                                  # <<----- create progress bar 
                for image, label in tqdm_epoch:
                    tqdm_epoch.set_description(f"EPOCH{epoch+1} / {EPOCH}")

                    #put image in cuda or cpu
                    image = image.to(DEVICE)
                    label = label.to(DEVICE)

                    #forward pass
                    outputs = loaded_model(image)
                    loss = criterion(outputs, label)
                    
                    # backward pros
                    optimizer.zero_grad()                     # <<--- clear past grad
                    loss.backward()                           # <<--- calculate gradient
                    optimizer.step()                          # <<--- update the weight
                    
                    tqdm_epoch.set_postfix(loss=loss.item())
                    
        logging.info(f"Model trained successfully")
        trained_model_path = os.path.join(config['artifacts']['model'], config['artifacts']['trained_model'])
        torch.save(loaded_model, trained_model_path)
        logging.info(f'trained model saved at {trained_model_path}')

    except Exception as e:
        logging.exception(e)
        print(e)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e