#!/bin/bash

# =======================
# Start Testing Pipeline
# =======================

cat <<EOF

=======================
Start Testing Pipeline
=======================

EOF

# set the config file path environment variable
CONFIG_FILE_PATH="$BASE_DIR/test_config.yml"

# if passed by argument --config-file then overwrite the config file path
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
  --config-file)
    CONFIG_FILE_PATH="$2"
    shift # past argument
    shift # past value
    ;;
  *)      # unknown option
    shift # past argument
    ;;
  esac
done

export CONFIG_FILE_PATH="$CONFIG_FILE_PATH"
echo "Used Config File: $CONFIG_FILE_PATH"

# =======================
# Config File Content
# =======================

cat <<EOF

=======================
Config File Content
=======================

EOF
cat "$CONFIG_FILE_PATH" | sed 's/^/  /'

# =========================
# Setting up Python Environment
# =========================

cat <<EOF


=======================
Setting up Python Environment
=======================

EOF

# check if environment variable USE_CONDA_ENVIRONMENT is set

echo "PYTHONPATH: $PYTHONPATH"

if [[ -z "${USE_CONDA_ENVIRONMENT}" ]]; then
  echo "USE_CONDA_ENVIRONMENT is not set, assume all dependencies are installed"
else
  # check if conda is installed
  if command -v conda &>/dev/null; then
    conda activate "${CONDA_ENVIRONMENT_NAME}"
    echo "conda environment '${CONDA_ENVIRONMENT_NAME}' activated"
  else
    echo "conda could not be found, assume all dependencies are installed"
  fi
fi

# =========================
# Start Testing
# =========================

cat <<EOF

=======================
Start Testing
=======================

EOF

# Set the signal handler function for USR1 signals
trap 'handle_signal USR1' USR1

if [[ -z "${RUNS_ON_EULER}" ]]; then
  python3 "$BASE_DIR/src/models/unet/test.py"
else

  # Define a signal handler function to forward USR1 signals to the Python process
  handle_signal() {
    if [ "$1" = "USR1" ]; then
      echo "Forwarding USR1 signal to Python process"
      echo "this is done using the following command: kill -USR1 $PYTHON_PID"
      kill -USR1 "$PYTHON_PID"
    fi
  }

  echo "RUN python '$BASE_DIR/models/unet/test.py'"

  python3 -u "$BASE_DIR/src/models/unet/test.py" \
    1>"$LOG_DIR/python_test_model.log" \
    2>"$LOG_DIR/python_test_model.error" &

  PYTHON_PID=$!

  # Wait for the Python process to finish
  echo "Waiting for Python process (PID=$PYTHON_PID) to finish"
  wait $PYTHON_PID
  wait # for some reason, the wait command does not wait for the Python process to finish, so we need to call it twice

fi
