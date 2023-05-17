FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV LANG en_US.UTF-8

RUN apt update

RUN apt install -y wget

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh

RUN sh Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b -p /opt/anaconda

ENV PATH="/opt/anaconda/bin:$PATH"

RUN pip install tensorflow geopandas pandas matplotlib rlxutils rasterio shapely wget wandb progressbar2 segmentation_models scikit-learn scikit-image seaborn

RUN pip install jupyterlab tensorrt tensorflow_probability

RUN echo 'alias jp="jupyter-lab --ServerApp.disable_check_xsrf=True --ip 0.0.0.0 --port 8811 --allow-root --NotebookApp.token='' --NotebookApp.password=''"' >> .bashrc

RUN apt install -y vim

RUN apt install -y libexpat-dev

RUN mkdir /opt/data

RUN ln -s /opt/anaconda/lib/python3.10/site-packages/tensorrt/libnvinfer.so.8 /opt/anaconda/lib/python3.10/site-packages/tensorrt/libnvinfer.so.7

RUN ln -s /opt/anaconda/lib/python3.10/site-packages/tensorrt/libnvinfer_plugin.so.8 /opt/anaconda/lib/python3.10/site-packages/tensorrt/libnvinfer_plugin.so.7

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/anaconda/lib/python3.10/site-packages/tensorrt"

ENV TF_CPP_MIN_LOG_LEVEL=1

CMD jupyter-lab --ServerApp.disable_check_xsrf=True --ip 0.0.0.0 --port 8811 --allow-root --NotebookApp.token='' --NotebookApp.password=''

