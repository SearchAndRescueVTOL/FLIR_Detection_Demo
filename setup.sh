mkdir ~/.kaggle 
touch ~/.kaggle/kaggle.json
sudo apt update
sudo apt install python3.12-venv
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt