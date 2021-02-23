set -e

pip install --upgrade setuptools wheel
pip install -r requirements.txt

wget https://storage.googleapis.com/dm-nfnets/F0_haiku.npz
