from importlib.resources import path
import logging
import os
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import argparse
import torch.nn.functional as F
from src import stage_01_get_data 
from src.utils.common import read_yaml, create_directories


STAGE = "stage_04_prediction"

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    config = read_yaml(config_path)
    test_data_batches = config['params']['no_of_test_data_batches_for_prediction']
    prediction_dir = Path(config['Data']['predicted_data'])
    create_directories([prediction_dir])

    prediction_file = Path(config['Data']['predicted_data'], config['Data']['prediction_file'])
    with open(prediction_file, 'w') as f:
        f.write('')
    trained_model_path = Path(config['artifacts']['model'], config['artifacts']['trained_model'])

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    train_data_loader, test_data_loader, label_map=  stage_01_get_data.main(config_path)
    #load model
    trained_model = torch.load(trained_model_path)
    # model switched in cuda
    trained_model.to(DEVICE)
    trained_model.eval()
    #get data
    c = 0
    count = 0
    with torch.no_grad():

        # for i in range(test_data_batches):        
        for images, lables in test_data_loader:
            if c >= test_data_batches:
                break
            c=c+1
            #load images in cuda
            images = images.to(DEVICE)
            lables = lables.to(DEVICE)
            # print(images.shape)
            logit = trained_model(images) # raw output
            actual = lables

        
            for raw_output, label in zip(logit, actual):
                count +=1
                probability = F.softmax(raw_output) # get the probability
                prediction = torch.argmax(probability)
                prediction_value = prediction.item() # get value from torch item
                cls = label_map[prediction_value]
                label = label.item() # get value from torch item
                acl_class = label_map[label]
                with open(prediction_file, 'a+') as f:
                    f.write(f"Predicted class is --> {cls} || Actual class is --> {acl_class}\n")
                    print(f"Predicted class is --> {cls} || Actual class is --> {acl_class}")
                    
            # c = c+1
        print(f"total count is {count}")
        logging.info(f"total count is {count}")


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
