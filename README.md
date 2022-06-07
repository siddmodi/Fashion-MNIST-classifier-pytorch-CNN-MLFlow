# Fashion-MNIST-classifier-pytorch-CNN-MLFlow

# STEPS

## Step 01 - Create a repository by using template repository
## step 02 - Clone the new repository
## step 03 - Create a conda environment after opening the repository in VSCODE
```
conda create --prefix ./env python=3.7 -y
```
## activate environment
```
conda activate ./env
```
### or
```
source activate ./env
```

## STEP 04- install the requirements
```
pip install -r requirements.txt
```
## step 05- install pytorch 11.3

```
--------------USE ANY ONE---------------------------------
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch    <<--- for cuda toolkit (GPU) , using conda 
conda install pytorch torchvision torchaudio cpuonly -c pytorch             <<--- for cpu , , using conda   
pip3 install torch torchvision torchaudio                                   <<--- pip installation
```
## or use init_setup.sh if not want run step 01 to step 05
### in bash terminal use below command
```
bash init_setup.sh
```
## step 06- install setup.py
```
pip install -e .
``` 

========================== Explaination ===========================

Flask app asking to upload picture and it'll predict out of 10 fashion product what it is.

We use MLops pipeline for this project to smoothen the process and seprate each stage from each other 

This project we divided in 5 stages. we use pytorch to define our architecture

1) To get fashion mnist data and use data loader class to load train and test images

2) Define a CNN archeitecture with convolutional filter ,Relu activation function and maxpooling layer ,Also create a forward pass layer.
 	 after creating a base model, we dump that model in respective folder

3) Here we train our base model with train data using Adam optimizer, also define backward pass layer save trained model in respective folder

4) Here we test our trained model with untrained data and tell what's its actual and predicted class

we create utils directory for all types of function such as some common , data management and for evaluating model. 
We also consider the situation if we stuck anywhere by creating log files for each and every steps for each stage

