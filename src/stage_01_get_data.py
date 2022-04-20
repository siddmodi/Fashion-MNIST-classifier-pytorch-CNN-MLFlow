import logging
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from src.utils.common import create_directories, read_yaml


STAGE = "stage_01_get_data"

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    config = read_yaml(config_path)
    data_folder_path = config['Data']['root_data_folder']
    create_directories([data_folder_path])         
    logging.info(f"getting data")

    ''' 1. In PyTorch, we mostly work with data in the form of tensors If the input data is in the form of a NumPy array 
            or PIL image,we can convert it into a tensor format using ToTensor 
        2. The final tensor will be of the form (C * H * W). Along with this, a scaling operation is also performed from 
            the range of 0-255 to 0-1.'''  
    
    train_data = datasets.FashionMNIST(root=data_folder_path, train=True, download=True,
                                        transform=transforms.ToTensor())
    test_data = datasets.FashionMNIST(root=data_folder_path, train=False, download=True, 
                                        transform= transforms.ToTensor())

    ''' The class_to_idx keeps track of our mapping of class values to the indices,
        where the indices are actually predicted by our model. The label_map can map the class back to
        the name of class '''

    given_label = train_data.class_to_idx
    logging.info(f'Labeling to each class : {given_label}')

    label_map = {val: key for key, val in given_label.items()}
    logging.info(f'Labeling to each class in order : {label_map}')

    logging.info(f"data is available at {data_folder_path}")

    logging.info(f"getting dataloader")

    '''
        DataLoader helps us to load the data in batches to pass it in our neural network
    '''
    train_data_loader = DataLoader(dataset = train_data,
                                    shuffle=True,
                                    batch_size=config['params']['BATCH_SIZE'])

    test_data_loader = DataLoader(dataset=test_data,
                                    shuffle=False,
                                    batch_size=config['params']['BATCH_SIZE'])
                                    
    return train_data_loader, test_data_loader, label_map


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()
    try:
        logging.info("\n********************")
        logging.info(f">>>>> {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e

