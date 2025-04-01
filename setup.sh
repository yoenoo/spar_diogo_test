set -e 

apt-get update && apt-get install -y vim

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$PATH:$HOME/.local/bin/env

uv init
uv venv --python=3.10
source .venv/bin/activate

uv pip install -r requirements.txt