image=pdf-document-convert:cann8.0.rc1-py39-torch1.11


: '
docker run -it --rm --cpus 16 -m 32G -h doc-converter --name doc-converter \
    --device /dev/davinci4 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ \
    -v /data/data01/myg/input:/workspace/input \
    -v /data/data01/myg/output:/workspace/output \
    -v /data/data01/myg/custom_convert.py:/workspace/custom_convert.py \
    -v /data/data01/myg/utils.py:/workspace/utils.py \
    -v /data/data01/myg/docx_chain:/workspace/docx_chain \
    -v /data/data01/myg/debug.sh:/workspace/debug.sh \
    $image bash
'

docker run -it --rm --cpus 16 -m 32G -h doc-converter --name doc-converter \
    --device /dev/davinci4 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ \
    -v /data/data01/myg/input:/workspace/input \
    -v /data/data01/myg/output:/workspace/output \
    $image bash