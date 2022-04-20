conda create --prefix ./env python=3.7 -y && source activate ./env 
pip install -r requirements.txt
# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch      
# conda install pytorch torchvision torchaudio cpuonly -c pytorch                 
pip3 install torch torchvision torchaudio                                       
conda env export > conda.yaml