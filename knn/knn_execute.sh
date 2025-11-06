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
LIBS=("scikit-learn" "matplotlib" "seaborn")
for i in {0..2}; do
  if [ $(pip list | grep -c ${LIBS[i]}) -lt 1 ]; then
    pip install ${LIBS[i]}
  fi
done

# Run program
python3 knntest.py

# Deactivate environment
deactivate
