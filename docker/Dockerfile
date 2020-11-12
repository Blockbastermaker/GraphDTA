#FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
#FROM nvidia/cuda:10.1-devel-ubuntu18.04 
#FROM meadml/cuda10.1-cudnn7-devel-ubuntu18.04-python3.6
#FROM docker.oa.com/seven-developer/bindingaffinity:v1.0-torch1.4-geometric-tf1.15-rdkit2019.03
#FROM docker.oa.com/g_zlzzheng/gromacs_openmm:base
#RUN rm -rf /root/anaconda3 && ls -lrt /root/
FROM docker.oa.com/g_tfplus/gromacs512_openmm:latest

# proxy
ENV http_proxy 'http://devnet-proxy.oa.com:8080'
ENV https_proxy 'http://devnet-proxy.oa.com:8080'


#RUN curl -fsSL https://mirrors.aliyun.com/nvidia-cuda/ubuntu1804/x86_64/7fa2af80.pub | apt-key add -
#RUN echo "deb https://mirrors.aliyun.com/nvidia-cuda/ubuntu1804/x86_64/ ./" > /etc/apt/sources.list.d/cuda.list

#RUN apt-get install -y --no-install-recommends wget

#RUN apt-get update && apt-get install -y --no-install-recommends vim bc wget

# change mirror
RUN    echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse" > /etc/apt/sources.list \
    && echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse" >> /etc/apt/sources.list

#ENV http_proxy  ''
#ENV https_proxy ''

#RUN wget -O /etc/apt/sources.list https://mirrors.tencent.com/doc/repo/ubuntu16_sources.list
#RUN wget -O /etc/apt/sources.list https://mirrors.tencent.com/doc/repo/ubuntu16_sources.list
#RUN apt-get update && apt-get install -y --no-install-recommends git
#RUN apt-get update #&& apt-get install -y --no-install-recommends vim bc

# basic functions
RUN apt-get update && apt-get install vim bc wget bzip2 ca-certificates libglib2.0-dev \
    libxext6 libsm6 libxrender1 \
    git mercurial subversion -y

# proxy
#ENV http_proxy 'http://devnet-proxy.oa.com:8080'
#ENV https_proxy 'https://devnet-proxy.oa.com:8080'

# anaconda 3
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# install pytorch tensorflow
RUN conda install -y -c conda-forge opencv && pip install tensorflow-gpu==1.15.0 && conda install -y -c rdkit rdkit=2019.03.1.0 \
    && conda install -y pytorch=1.4 torchvision cudatoolkit -c pytorch && pip install torch-scatter==latest+cu101 \
    -f https://pytorch-geometric.com/whl/torch-1.4.0.html && pip install torch-sparse==latest+cu101 \
    -f https://pytorch-geometric.com/whl/torch-1.4.0.html && pip install torch-cluster==latest+cu101 \
    -f https://pytorch-geometric.com/whl/torch-1.4.0.html && pip install torch-spline-conv==latest+cu101 \
    -f https://pytorch-geometric.com/whl/torch-1.4.0.html && pip install torch-geometric

