from distutils.command.config import config
from unittest import result
from urllib.error import ContentTooShortError
import torch
import os
import logging
from src.utils.common import read_yaml, create_directories
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
from src import stage_01_get_data


STAGE = "evaluation_model"

logging.basicConfig(filename=os.path.join('logs', 'running_logs.log'),
                    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
                    filemode="a"
)

def main(config_path):
    content = read_yaml(config_path)
    trained_model_path = Path(content['artifacts']['model'], content['artifacts']['trained_model'])
    trained_model = torch.load(trained_model_path)
    trained_model.eval()
    DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
    train_data_loader, test_data_loader, label_map = stage_01_get_data.main(config_path)
    test_batch = content['evaluation']['no_of_test_batches']
    no_of_test_batches = test_batch ## 1 batch contain batch_size images
    # LOAD DATA FROM DATALOADER
    
    pred = np.array([])
    actual = np.array([])
    with torch.no_grad():

        for i in range(no_of_test_batches):

            for images, labels in test_data_loader:
                ##put images and labels in cuda
                images = images.to(DEVICE) # it will contain 32 images as default batchsize=32
                # labels = labels.to(DEVICE)

                raw_prediction = trained_model(images) # it will predict the class of images 
                prediction = torch.argmax(raw_prediction, dim=1) # give the class have max probability

                y_pred = prediction.cpu().numpy() # load in cpu and convert it numpy from torch

                pred = np.concatenate((pred,y_pred)) # concatenate data to numpy array
                actual = np.concatenate((actual, labels))
    cm = confusion_matrix(actual, pred)
    logging.info(f"confusion matrix is {cm}")

    # plot confusion matrix
    fig_path = Path(content['artifacts']['model'], content['artifacts']['confusion_matrix_fig'])
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True,fmt='d', cbar=False, xticklabels=label_map.values(), yticklabels=label_map.values())
    result = fig_path
    plt.savefig(result)
    plt.show()
    logging.info(f"confusion matrix figure is saved at {result}")
    

if __name__ == '__main__':
    logging.info("\n**********************")
    logging.info(f">>>>>>>>>{STAGE} started<<<<<<<<<")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config/config.yaml")
    parsed_arg = parser.parse_args()
    try:
        main(parsed_arg.config)
        logging.info(f"{STAGE} completed successfully")
    except Exception as e:
        logging.exception(e)
        print(e)
        raise e