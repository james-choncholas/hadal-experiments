export DEBIAN_FRONTEND="noninteractive"

sudo apt update
sudo apt upgrade -y
sudo apt install -y sudo git git-lfs unzip python3 python3-pip python3-venv


python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
