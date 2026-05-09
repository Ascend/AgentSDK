FROM quay.io/ascend/verl:verl-8.3.rc1-910b-ubuntu22.04-py3.11-v0.7.0

SHELL ["/bin/bash", "-c"]

# =========================
# System dependencies
# =========================
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        openssh-server \
        dos2unix \
        git \
        wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# =========================
# Python dependencies
# =========================
RUN pip install --no-cache-dir \
    vertexai \
    sentence_transformers \
    latex2sympy2 \
    word2number \
    mlflow

# =========================
# Apply verl patch
# =========================
RUN wget -P /verl https://raw.githubusercontent.com/rllm-org/rllm/b5b9760fc3d7208d368f21dcd3e12f4f7eddfdc7/rllm/experimental/fully_async/verl_dp_actor.patch && \
    cd /verl && \
    patch -p1 < /verl/verl_dp_actor.patch

# =========================
# Clone AgentSDK repository
# =========================
RUN mkdir -p /home/work && \
    cd /home/work && \
    git clone https://gitcode.com/Ascend/AgentSDK.git

# =========================
# Install third-party dependencies
# =========================
RUN cd /home/work/AgentSDK && \
    bash download_third_party.sh

# =========================
# Environment variables
# =========================
ENV VLLM_ASCEND_ENABLE_NZ=0

# =========================
# Hosts configuration (build-time)
# =========================
RUN echo "127.0.0.1 $(hostname)" >> /etc/hosts

# =========================
# Default command
# =========================
CMD ["/bin/bash"]
