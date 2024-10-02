conda install nvidia/label/cuda-12.1.0::cuda-nvcc -y
conda install nvidia/label/cuda-12.1.0::cuda-toolkit -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r ./environment/requirements.txt
cd diffusers
pip install -e .
cd ..
