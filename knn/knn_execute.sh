: '
Set up environment and execute knn code
'


# Install venv
if [ $(ls -a .. | grep -c ".venv") -lt 1 ]; then
  python3 -m venv ../.venv
fi

# Activate venv
source ../.venv/bin/activate

# Install libraries
if [ $(pip list | grep -c "scikit-learn") -lt 1]; then
  pip install scikit-learn
fi

# Run program
python3 knntest.py

# Deactivate environment
deactivate