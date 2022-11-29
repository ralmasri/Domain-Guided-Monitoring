FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ARG CONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
ARG CONDA_SH=Miniconda3-py38_4.12.0-Linux-x86_64.sh
ARG CONDA_DIR=.tmp

RUN mkdir /Domain-Guided-Monitoring
RUN mkdir /Domain-Guided-Monitoring/artifacts
RUN mkdir /Domain-Guided-Monitoring/mlruns

COPY data /Domain-Guided-Monitoring/data
COPY notebooks /Domain-Guided-Monitoring/notebooks
COPY src /Domain-Guided-Monitoring/src
COPY tests /Domain-Guided-Monitoring/tests
COPY environment.yml /Domain-Guided-Monitoring
COPY r_install.sh /Domain-Guided-Monitoring
COPY main_refinement.py /Domain-Guided-Monitoring
COPY main.py /Domain-Guided-Monitoring
COPY Makefile /Domain-Guided-Monitoring

# Causality algorithms need GSL >=2.5
RUN apt-get update && \
    apt install software-properties-common -y && \ 
    apt-get update && \
    add-apt-repository -y ppa:dns/gnu && \ 
    apt-get update && \
    apt install libgsl-dev -y

# DomainML and its dependencies installation
RUN cd /Domain-Guided-Monitoring && \
    mkdir .tmp && \
    apt-get install wget -y && \
    apt-get install nano -y && \
    wget -nc -P .tmp/ $CONDA_URL && \
    bash ./$CONDA_DIR/$CONDA_SH -b -u -p ./$CONDA_DIR/miniconda3/ && \
    .tmp/miniconda3/bin/conda env update --file environment.yml --prune && \
    .tmp/miniconda3/bin/conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch && \
    bash r_install.sh