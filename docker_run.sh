IMAGE=document-convert:ubuntu20.04-cuda11.7.1-py38-torch2.0.1-tf1.15.5-1.8.1
DEVICE=3

docker run -it --rm --gpus "device=$DEVICE" -h doc-convert --name doc-convert \
    -v /data/lm/github/document-convert/input:/workspace/input \
    -v /data/lm/github/document-convert/output:/workspace/output \
    -v /data/lm/github/document-convert/custom_convert.py:/workspace/custom_convert.py \
    -v /data/lm/github/document-convert/utils.py:/workspace/utils.py \
    $IMAGE bash
