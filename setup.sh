set -e 

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$PATH:$HOME/.local/bin/env

# $HOME/.local/bin/uv venv --python=3.10
uv venv --python=3.10
source .venv/bin/activate
uv pip install -r requirements.txt