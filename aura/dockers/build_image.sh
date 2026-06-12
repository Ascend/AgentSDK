#!/bin/bash
set -e

DMI_FILE="/sys/class/dmi/id/product_name"
DOCKERFILE_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "$DMI_FILE" ]; then
    echo "错误：无法读取 $DMI_FILE，无法判断服务器类型"
    exit 1
fi

PRODUCT_NAME=$(cat "$DMI_FILE")
echo "检测到服务器型号：$PRODUCT_NAME"

if echo "$PRODUCT_NAME" | grep -q "A3"; then
    SERVER_TYPE="A3"
    BASE_IMAGE="swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-a3-ubuntu22.04-py3.11"
    SOC_VERSION="ascend910_9391"
    echo "识别为 Atlas A3 服务器"
elif echo "$PRODUCT_NAME" | grep -q "A2"; then
    SERVER_TYPE="A2"
    BASE_IMAGE="swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-910b-ubuntu22.04-py3.11"
    SOC_VERSION="ascend910b1"
    echo "识别为 Atlas A2 服务器"
else
    echo "错误：不支持当前服务器类型：$PRODUCT_NAME"
    echo "仅支持 Atlas A2 和 Atlas A3 服务器"
    exit 1
fi

IMAGE_NAME="aura-${SERVER_TYPE,,}:latest"

echo ""
echo "构建参数："
echo "  BASE_IMAGE  : $BASE_IMAGE"
echo "  SOC_VERSION : $SOC_VERSION"
echo "  IMAGE_NAME  : $IMAGE_NAME"
echo ""

cd "$DOCKERFILE_DIR"

docker build \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg SOC_VERSION="$SOC_VERSION" \
    -t "$IMAGE_NAME" \
    .

echo ""
echo "构建完成，镜像名称：$IMAGE_NAME"
