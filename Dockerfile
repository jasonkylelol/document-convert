# https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/huawei_ascend.md#%E4%BD%BF%E7%94%A8docker%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83
from paddle-npu-base:py37-cann8_0_rc1-aarch64

WORKDIR /workspace

RUN apt-get update && apt-get install -y vim libsm6 libxext6 libxrender-dev libgl1-mesa-glx ffmpeg

add github/FastDeploy/python/dist/* /build/dist/
add github/pytorch/dist/* /build/dist/

COPY github/document-convert/requirements.txt requirements.txt
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip setuptools wheel
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt 
run pip install /build/dist/* jinja2 absl-py cloudpickle tornado

add /workspace/FastDeploy/python/fastdeploy/libs/third_libs/opencv/lib/ /usr/local/fastdeploy/libs/third_libs/opencv/lib
run export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/fastdeploy/libs/third_libs/opencv/lib

run rm -rf /build