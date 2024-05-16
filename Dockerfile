FROM registry.baidubce.com/paddlepaddle/paddle:2.5.1-gpu-cuda11.2-cudnn8.2-trt8.0
# python=3.7.13
WORKDIR /workspace

RUN apt-get update && apt-get install -y vim libsm6 libxext6 libxrender-dev libgl1-mesa-glx ffmpeg 
RUN /usr/bin/python -V

# RUN pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html -i https://pypi.tuna.tsinghua.edu.cn/simple
# https://bj.bcebos.com/fastdeploy/release/wheels/fastdeploy_gpu_python-1.0.7-cp37-cp37m-manylinux1_x86_64.whl
COPY build/fastdeploy_gpu_python-1.0.7-cp37-cp37m-manylinux1_x86_64.whl /wheels/
RUN pip install /wheels/fastdeploy_gpu_python-1.0.7-cp37-cp37m-manylinux1_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple

# https://download.pytorch.org/whl/cu111/torch-1.10.1%2Bcu111-cp37-cp37m-linux_x86_64.whl
COPY build/torch-1.10.1+cu111-cp37-cp37m-linux_x86_64.whl /wheels/
RUN pip install /wheels/torch-1.10.1+cu111-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple

# https://download.pytorch.org/whl/cu111/torchvision-0.11.2%2Bcu111-cp37-cp37m-linux_x86_64.whl
COPY build/torchvision-0.11.2+cu111-cp37-cp37m-linux_x86_64.whl /wheels/
RUN pip install /wheels/torchvision-0.11.2+cu111-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install timm==0.5.4 --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN rm -rf /wheels/

ADD models models
ADD cls cls
ADD det det
ADD latex latex
ADD ppocr ppocr
ADD rec rec
ADD struc struc
ADD table table
ADD tools tools
ADD config.yaml config.yaml
ADD multi_thread_process_to_doc.py multi_thread_process_to_doc.py
ADD utils.py utils.py
