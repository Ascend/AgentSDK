FROM eulers:latest AS builder
SHELL ["/bin/bash", "-c"]
USER root

# Copy all the required files to a temporary directory at once (to avoid scattered COPY instructions)
COPY miniconda/Miniconda3-py310_25.5.1-1-Linux-aarch64.sh /tmp/
COPY cann/Ascend-*.run /tmp/cann/
COPY PTA/*.whl /tmp/pta/
COPY rm_cert.sh /tmp/

RUN ls -la /tmp

COPY modelverse_probe.tar /home/ModelVerse/
COPY pre_install.sh /home/install/
COPY aura.tar.gz /home/Aura/
COPY install_vllm_mindspeed.sh /home/
COPY init.sh /usr/local/bin/init.sh

RUN rm -rf /etc/pki/ca-trust/extracted/openssl/ca-bundle.trust.crt && \
    rm -rf /etc/pki/rpm-gpg/RPM-GPG-KEY-EulerOS && \
    rm -rf /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem


FROM scratch
COPY --from=builder / /

RUN chmod +x /usr/local/bin/init.sh && \
    # === Step 1: Configure yum repo === \
    echo '[openEuler-everything]' > /etc/yum.repos.d/openeuler.repo && \
    echo 'name=openEuler-everything' >> /etc/yum.repos.d/openeuler.repo && \
    echo 'baseurl=http://mirrors.tools.huawei.com/openeuler/openEuler-22.03-LTS-SP4/everything/aarch64/' >> /etc/yum.repos.d/openeuler.repo && \
    echo 'enabled=1' >> /etc/yum.repos.d/openeuler.repo && \
    echo 'gpgcheck=0' >> /etc/yum.repos.d/openeuler.repo && \
    echo '[openEuler-EPOL]' >> /etc/yum.repos.d/openeuler.repo && \
    echo 'name=openEuler-epol' >> /etc/yum.repos.d/openeuler.repo && \
    echo 'baseurl=http://mirrors.tools.huawei.com/openeuler/openEuler-22.03-LTS-SP4/EPOL/main/aarch64/' >> /etc/yum.repos.d/openeuler.repo && \
    echo 'enabled=1' >> /etc/yum.repos.d/openeuler.repo && \
    echo 'gpgcheck=0' >> /etc/yum.repos.d/openeuler.repo && \
    echo '[openEuler-update]' >> /etc/yum.repos.d/openeuler.repo && \
    echo 'name=openEuler-update' >> /etc/yum.repos.d/openeuler.repo && \
    echo 'baseurl=http://mirrors.tools.huawei.com/openeuler/openEuler-22.03-LTS-SP4/update/aarch64/' >> /etc/yum.repos.d/openeuler.repo && \
    echo 'enabled=1' >> /etc/yum.repos.d/openeuler.repo && \
    echo 'gpgcheck=0' >> /etc/yum.repos.d/openeuler.repo && \
    \
    # === Step 2: Configure hosts & Install basic tools === \
    echo '7.223.199.227 mirrors.tools.huawei.com' >> /etc/hosts && \
    echo '7.223.217.116 cmc-cd-mirror.rnd.huawei.com' >> /etc/hosts && \
    yum update -y && \
    yum install -y vim git gcc g++ cmake wget pciutils net-tools kmod systemd systemd-udev && \
    yum groupinstall "Development Tools" -y && \
    \
    # === Step 3: Install Miniconda === \
    mkdir -p /home/miniconda && \
    mv /tmp/Miniconda3-py310_25.5.1-1-Linux-aarch64.sh /home/miniconda/ && \
    bash /home/miniconda/Miniconda3-py310_25.5.1-1-Linux-aarch64.sh -b -p /home/miniconda3 && \
    /home/miniconda3/bin/conda init bash && \
    export PATH="/home/miniconda3/bin:$PATH" && \
    eval "$(/home/miniconda3/bin/conda shell.bash hook)" && \
    \
    # === Step 4: Configure pip === \
    mkdir -p /root/.pip && \
    echo '[global]' > /root/.pip/pip.conf && \
    echo 'timeout = 60' >> /root/.pip/pip.conf && \
    echo 'trusted-host = mirrors.tools.huawei.com' >> /root/.pip/pip.conf && \
    echo 'index-url = http://mirrors.tools.huawei.com/pypi/simple' >> /root/.pip/pip.conf && \
    pip config set global.trusted-host cmc-cd-mirror.rnd.huawei.com && \
    pip config set global.index-url http://cmc-cd-mirror.rnd.huawei.com/pypi/simple/ && \
    pip install torch==2.7.1 && \
    pip install ray==2.48.0 && \
    \
    # === Step 5: Install CANN === \
    mkdir -p /opt/packages/cann && \
    mv /tmp/cann/*.run /opt/packages/cann/ && \
    chmod +x /opt/packages/cann/*.run && \
    groupadd HwHiAiUser && \
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash && \
    cd /opt/packages/cann && \
    ./Ascend-cann-toolkit_*_linux-aarch64.run --full --quiet && \
    ./Ascend-cann-kernels-910b_*_linux-aarch64.run --install --quiet && \
    source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    ./Ascend-cann-nnal_*_linux-aarch64.run --install --quiet && \
    \
    # === Step 6: Install PTA (PyTorch + torch_npu) === \
    mkdir -p /opt/packages/PTA && \
    mv /tmp/pta/*.whl /opt/packages/PTA/ && \
    cd /opt/packages/PTA && \
    pip install torch_npu-*.whl && \
    \
    # === Step 7: Install vLLM=== \
    cd /home && \
    chmod +x install_vllm_mindspeed.sh && \
    ./install_vllm_mindspeed.sh && \
    rm install_vllm_mindspeed.sh && \
    \
    # === Step 8: Reinstall torch/torch_npu \
    pip install torch==2.7.1 && \
    cd /opt/packages/PTA && \
    pip install torch_npu-*.whl && \
    # === Step 9: Clean up sensitive files and temporary content ===
    rm -rf /opt/packages/cann/*.run && \
    rm -rf /opt/packages/PTA/*.whl && \
    rm -f /home/miniconda/Miniconda3-py310_25.5.1-1-Linux-aarch64.sh && \
    # Optional: Clean yum cache
    yum clean all && \
    # === Step 10: Execute the custom certificate cleanup script ===
    chmod +x /tmp/rm_cert.sh && \
    /tmp/rm_cert.sh && \
    rm -f /tmp/rm_cert.sh && \
    rm -rf /tmp/* && \
    rm -rf /home/recover/pip-23.0.1-py3-none-any.whl && \
    rm -rf /etc/pki/ca-trust/extracted/openssl/ca-bundle.trust.crt && \
    rm -rf /etc/pki/rpm-gpg/RPM-GPG-KEY-EulerOS && \
    rm -rf /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem && \
    # === Finally clean up temporary entries in hosts (optional, proceed with caution)===
    echo "========sed====================" && \
    echo "=============sed end=============="


# Note: Since all installations are completed in a single RUN, the PATH needs to be reset at runtime
ENV PATH="/home/miniconda3/bin:$PATH"

ENTRYPOINT ["/usr/local/bin/init.sh"]
CMD ["tail", "-f", "/dev/null"]
