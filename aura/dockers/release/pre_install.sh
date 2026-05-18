#!/bin/bash

rm -rf /home/miniconda3/lib/python3.10/site-packages/pip
rm -rf /home/miniconda3/lib/python3.10/site-packages/pip-*.dist-info

export no_proxy="mirrors.tools.huawei.com"
wget --no-check-certificate https://mirrors.tools.huawei.com/pypi/packages/07/51/2c0959c5adf988c44d9e1e0d940f5b074516ecc87e96b1af25f59de9ba38/pip-23.0.1-py3-none-any.whl

cp pip-23.0.1-py3-none-any.whl /home/miniconda3/lib/python3.10/ensurepip/_bundled/
cp pip-23.0.1-py3-none-any.whl /home/miniconda3/pkgs/python-3.10.18-hbb0f47a_0/lib/python3.10/ensurepip/_bundled/

python3 -m ensurepip --upgrade

read -p "proxy user name: " proxy_user
read -s -p "proxy passward " proxy_pass
echo ""

export http_proxy="http://${proxy_user}:${proxy_pass}@proxycn.huawei.com:8080"
export https_proxy=${http_proxy}

wget --no-check-certificate https://mirrors.aliyun.com/pypi/packages/e0/f0/8a2806114cd36e282823fd4d8e88e3b94dc943c2569c350d0c826a49db38/pip-25.1-py3-none-any.whl
pip install --no-index --find-links=./ pip-25.1-py3-none-any.whl

wget --no-check-certificate https://mirrors.aliyun.com/pypi/packages/4f/52/34c6cf5bb9285074dc3531c437b3919e825d976fde097a7a73f79e726d03/certifi-2025.7.14-py3-none-any.whl
wget --no-check-certificate https://mirrors.aliyun.com/pypi/packages/7c/e4/56027c4a6b4ae70ca9de302488c5ca95ad4a39e190093d6c1a8ace08341b/requests-2.32.4-py3-none-any.whl
wget --no-check-certificate https://mirrors.aliyun.com/pypi/packages/a7/c2/fe1e52489ae3122415c51f387e221dd0773709bad6c6cdaa599e8a2c5185/urllib3-2.5.0-py3-none-any.whl
wget --no-check-certificate https://mirrors.aliyun.com/pypi/packages/ae/d1/164e3c9d559133a38279215c712b8ba38e77735d3412f37711b9f8f6f7e0/cryptography-45.0.3-cp37-abi3-manylinux_2_34_aarch64.whl
wget --no-check-certificate https://mirrors.aliyun.com/pypi/packages/d5/94/d67756638d7bb07750b07d0826c68e414124574b53840ba1ff777abcd388/grpcio-1.74.0-cp310-cp310-manylinux_2_17_aarch64.whl
wget --no-check-certificate https://mirrors.aliyun.com/pypi/packages/5c/af/1f9d7f7faafe2ddfb6f72a2e07a548a629c61ad510fe60f9630309908fef/charset_normalizer-3.4.4-cp310-cp310-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64.whl
wget --no-check-certificate https://mirrors.aliyun.com/pypi/packages/0e/61/66938bbb5fc52dbdf84594873d5b51fb1f7c7794e9c0f5bd885f30bc507b/idna-3.11-py3-none-any.whl
wget --no-check-certificate https://mirrors.aliyun.com/pypi/packages/4f/27/6933a8b2562d7bd1fb595074cf99cc81fc3789f6a6c05cdabb46284a3188/cffi-2.0.0-cp310-cp310-manylinux2014_aarch64.manylinux_2_17_aarch64.whl
wget --no-check-certificate https://mirrors.aliyun.com/pypi/packages/a0/e3/59cd50310fc9b59512193629e1984c1f95e5c8ae6e5d8c69532ccc65a7fe/pycparser-2.23-py3-none-any.whl
pip install --force-reinstall --no-index --find-links=./ certifi-2025.7.14-py3-none-any.whl
pip install --force-reinstall --no-index --find-links=./ requests-2.32.4-py3-none-any.whl
pip install --force-reinstall --no-index --find-links=./ urllib3-2.5.0-py3-none-any.whl
pip install --force-reinstall --no-index --find-links=./ cryptography-45.0.3-cp37-abi3-manylinux_2_34_aarch64.whl
pip install --force-reinstall --no-index --find-links=./ grpcio-1.74.0-cp310-cp310-manylinux_2_17_aarch64.whl
