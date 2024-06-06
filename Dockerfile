from paddle-npu:cann80RC1-ubuntu20-aarch64-gcc84-py310-dev

WORKDIR /workspace

RUN apt-get update && apt-get install -y vim libsm6 libxext6 libxrender-dev libgl1-mesa-glx ffmpeg
RUN /usr/bin/python -V

run echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/fastdeploy/libs/third_libs/opencv/lib' >> /root/.bashrc

COPY requirements.txt requirements.txt
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --force-reinstall torch==2.1.0 torchvision torch-npu==2.1.0.post3

ADD build/Miniconda3-latest-Linux-aarch64.sh /root/miniconda3/miniconda_install.sh
RUN bash /root/miniconda3/miniconda_install.sh -b -u -p /root/miniconda3
RUN /root/miniconda3/bin/conda init bash
RUN /root/miniconda3/bin/conda install -y -c tartansandal conda-bash-completion
RUN rm -f /root/miniconda3/miniconda_install.sh
COPY build/condarc /root/.condarc
RUN /root/miniconda3/bin/conda create -y -n langchain python=3.11

ADD models models
ADD cls cls
ADD det det
ADD latex latex
ADD ppocr ppocr
ADD rec rec
ADD struc struc
ADD table table
ADD tools tools
ADD docx_chain docx_chain
ADD config.yaml config.yaml
ADD custom_convert.py custom_convert.py
ADD utils.py utils.py