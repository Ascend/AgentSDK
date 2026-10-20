#!/usr/bin/env bash
set -e

cd /home/work/AgentSDK/aura
bash download_third_party.sh
pip install -r third_party/requirements_aura.txt
pip uninstall -y pyarrow
pip install pyarrow==24.0.0
