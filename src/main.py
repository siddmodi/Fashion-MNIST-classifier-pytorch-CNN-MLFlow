import os
import logging
from matplotlib.style import use
import mlflow

from src.utils.common import read_yaml, create_directories

STAGE = "MAIN"

create_directories(["logs"])
logging.basicConfig(filename=os.path.join("logs", "running_logs.log"),
                    level=logging.INFO,
                    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
                    filemode="w+"
                    )

def main():
    with mlflow.start_run() as run:
        # mlflow.run(".", "get_data", use_conda=False)
        mlflow.run(".", "base_model_creation", use_conda=False)
        mlflow.run('.', "training_model", use_conda=False)
        mlflow.run('.', "prediction", use_conda=False)


if __name__ == '__main__':

    try:
        logging.info(f"*****************************")
        logging.info(f">>>>>>>>>>>>>>>>>STAGE: {STAGE} STARTED<<<<<<<<<<<<<<<<<<<<<")
        main()
        # logging.info(">>>>>>>>>>>>>>>>>STAGE: {STAGE} COMPLETED<<<<<<<<<<<<<<<<<<<<<")
    except  Exception as e:
        print(e)
        logging.exception(e)
        raise e

        