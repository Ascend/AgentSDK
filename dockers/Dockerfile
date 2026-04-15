FROM euleros:latest
SHELL ["/bin/bash", "-c"]
USER root

# construct yum environment
RUN echo '[openEuler-everything]' > /etc/yum.repos.d/openeuler.repo && \
    echo 'name=openEuler-everything' >> /etc/yum.repos.d/openeuler.repo && \
    echo 'baseurl=http://mirrors.tools.huawei.com/openeuler/openEuler-22.03-LTS-SP4/everything/aarch64/' >> /etc/yum.repos.d/openeuler.repo && \
    echo 'enabled=1' >> /etc/yum.repos.d/openeuler.repo && \
    echo 'gpgcheck=0' >> /etc/yum.repos.d/openeuler.repo && \
    echo 'gpgkey=http://mirrors.tools.huawei.com/openeuler/openEuler-22.03-LTS-SP4/everything/aarch64/RPM-GPG-KEY-openEuler' >> /etc/yum.repos.d/openeuler.repo && \
    echo '' >> /etc/yum.repos.d/openeuler.repo && \
    echo '[openEuler-EPOL]' >> /etc/yum.repos.d/openeuler.repo && \
    echo 'name=openEuler-epol' >> /etc/yum.repos.d/openeuler.repo && \
    echo 'baseurl=http://mirrors.tools.huawei.com/openeuler/openEuler-22.03-LTS-SP4/EPOL/main/aarch64/' >> /etc/yum.repos.d/openeuler.repo && \
    echo 'enabled=1' >> /etc/yum.repos.d/openeuler.repo && \
    echo 'gpgcheck=0' >> /etc/yum.repos.d/openeuler.repo && \
    echo '' >> /etc/yum.repos.d/openeuler.repo && \
    echo '[openEuler-update]' >> /etc/yum.repos.d/openeuler.repo && \
    echo 'name=openEuler-update' >> /etc/yum.repos.d/openeuler.repo && \
    echo 'baseurl=http://mirrors.tools.huawei.com/openeuler/openEuler-22.03-LTS-SP4/update/aarch64/' >> /etc/yum.repos.d/openeuler.repo && \
    echo 'enabled=1' >> /etc/yum.repos.d/openeuler.repo && \
    echo 'gpgcheck=0' >> /etc/yum.repos.d/openeuler.repo

RUN echo '7.223.199.227 mirrors.tools.huawei.com' >> /etc/hosts && \
    yum update -y && \
    yum install -y vim git gcc g++ cmake wget && \
    yum install -y pciutils net-tools kmod systemd systemd-udev && \
    yum groupinstall "Development Tools" -y && \
    yum clean all

# construct python env
WORKDIR /home/miniconda
COPY miniconda/Miniconda3-py310_25.5.1-1-Linux-aarch64.sh .
RUN bash Miniconda3-py310_25.5.1-1-Linux-aarch64.sh -b -p /home/miniconda3 && \
    /home/miniconda3/bin/conda init bash
ENV PATH="/home/miniconda3/bin:$PATH"

RUN mkdir -p /root/.pip && \
    echo '[global]' > /root/.pip/pip.conf && \
    echo 'timeout = 60' >> /root/.pip/pip.conf && \
    echo 'trusted-host = mirrors.tools.huawei.com' >> /root/.pip/pip.conf && \
    echo 'index-url = http://mirrors.tools.huawei.com/pypi/simple' >> /root/.pip/pip.conf


# set pip procy
RUN echo '7.223.217.116 cmc-cd-mirror.rnd.huawei.com' >> /etc/hosts && \
    pip config set global.trusted-host cmc-cd-mirror.rnd.huawei.com && \
    pip config set global.index-url http://cmc-cd-mirror.rnd.huawei.com/pypi/simple/ && \
    pip install ray[default]

# construct cann environment
WORKDIR /opt/packages/cann

ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest

COPY cann/Ascend-cann-kernels-910b_8.2.RC1_linux-aarch64.run .
COPY cann/Ascend-cann-nnal_8.2.RC1_linux-aarch64.run .
COPY cann/Ascend-cann-toolkit_8.2.RC1_linux-aarch64.run .
COPY cann/Ascend-hdk-910b-npu-driver_25.2.0_linux-aarch64.run .
COPY cann/Ascend-hdk-910b-npu-firmware_7.7.0.6.236.run .


RUN chmod +x *.run && \
    groupadd HwHiAiUser && \
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash && \
    eval "$(/home/miniconda3/bin/conda shell.bash hook)" && \
    ./Ascend-cann-toolkit_8.2.RC1_linux-aarch64.run --full --quiet && \
    ./Ascend-cann-kernels-910b_8.2.RC1_linux-aarch64.run --install --quiet && \
    source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    ./Ascend-cann-nnal_8.2.RC1_linux-aarch64.run --install --quiet

# construct PTA
# wget requires proxy configuration; it is recommended to download first and then copy it in.
# RUN wget https://download.pytorch.org/whl/cpu/torch-2.5.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl && \
#     eval "$(/home/miniconda3/bin/conda shell.bash hook)" && \
#     pip install torch-2.5.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
# RUN wget https://gitee.com/ascend/pytorch/releases/download/v7.1.0-pytorch2.5.1/torch_npu-2.5.1.post1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl && \
#     eval "$(/home/miniconda3/bin/conda shell.bash hook)" && \
#     pip install torch_npu-2.5.1.post1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
WORKDIR /opt/packages/PTA
COPY PTA/torch-2.5.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl .
COPY PTA/torch_npu-2.5.1.post1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl .

RUN echo '7.223.217.116 cmc-cd-mirror.rnd.huawei.com' >> /etc/hosts && \
    eval "$(/home/miniconda3/bin/conda shell.bash hook)" && \
    pip install torch-2.5.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl && \
    pip install torch_npu-2.5.1.post1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# construct vllm env
# github.com requires proxy configuration; it is recommended to download first and then copy it in.
# RUN git clone -b releases/v0.9.1 https://github.com/vllm-project/vllm.git && \
#     cd vllm && \
#     git checkout b6553be1bc75f046b00046a4ad7576364d03c835 && \
#     VLLM_TARGET_DEVICE=empty pip install . && \
#     cd ..
# RUN git clone -b v0.9.1-dev https://github.com/vllm-project/vllm-ascend.git && \
#     cd vllm-ascend && \
#     pip install -r requirements.txt && \
#     pip install -e .
WORKDIR /home
COPY install_vllm_mindspeed.sh ./install_vllm_mindspeed.sh
COPY init.sh /usr/local/bin/init.sh
RUN chmod +x /usr/local/bin/init.sh
ENTRYPOINT ["/usr/local/bin/init.sh"]
CMD ["tail", "-f", "/dev/null"]