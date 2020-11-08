#conda create -n graphdta python=3.6 tensorflow-gpu=1.15 pytorch
#conda activate graphdta
apt insttall vim bc 
apt install libxetx-ddev
conda install -y -c conda-forge opencv
pip install tensorflow-gpu==1.15.0
conda install -y -c rdkit rdkit=2019.03.1.0
conda install pytorch=1.4 torchvision cudatoolkit -c pytorch

pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-geometric

