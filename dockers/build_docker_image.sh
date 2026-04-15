IMAGE_NAME="agentic-rl:1.0.0"
IMAGE_NAME_TMP="agentic-rl:1.0.0.temp"
DOCKER_NAME_TMP="agentic_rl_tmp"
VLLM_ASCEND_PATH="/home/vllm-ascend"

docker build --network=host --force-rm=true -t $IMAGE_NAME_TMP .

docker stop $DOCKER_NAME_TMP
docker rm $DOCKER_NAME_TMP

docker run -itd --privileged=true --ipc=host --net=host \
    --name $DOCKER_NAME_TMP \
    --device=/dev/davinci0 --device=/dev/davinci1 \
    --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 \
    --device=/dev/davinci6 --device=/dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    $IMAGE_NAME_TMP

docker exec -it $DOCKER_NAME_TMP bash

# 以下命令在容器内手动执行（需要在交互shell环境中安装）
# cd /home
# bash install_vllm_mindspeed.sh
# exit # 退出自动执行后续操作

docker commit $DOCKER_NAME_TMP $IMAGE_NAME