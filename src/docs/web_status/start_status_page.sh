#!/usr/bin/env bash
# start_status_page.sh
set -e
echo "  f4e6  [1;32m安装依赖（仅首次） [0m"
pip install -r web_status/requirements.txt -q
echo "  f680  [1;34m启动状态监控页面... [0m"
cd web_status
python status_server.py 