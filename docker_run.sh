image=pdf-document-convert:v0.1
gpus="device=3"

docker run -it --rm --cpus 16 -m 32G --gpus $gpus -h doc-converter --name doc-converter \
    -v /home/lm/github/document-convert/input:/workspace/input \
    -v /home/lm/github/document-convert/output:/workspace/output \
    $image bash

# docker run -it --rm --cpus 16 -m 32G --gpus $gpus -h doc-converter --name doc-converter \
#     -v /home/lm/github/document-convert:/workspace \
#     -v /data/lm/huggingface/models/breezedeus/cnstd-cnocr-models/models/cnstd:/root/.cnstd \
#     -v /home/lm/github/AdvancedLiterateMachinery:/root/AdvancedLiterateMachinery \
#     $image bash
