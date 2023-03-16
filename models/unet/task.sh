echo "All further logs are saved in $LOG_DIR/python.log"

# run the actual task
python main.py --retrain >>"$LOG_DIR/python.log" &

# trap om USR1, SIGTERM, SIGKILL, and SIGINT and echo the signal id
# this is used to save the model before the slurm job is killed
trap 'echo "Received USR1 signal, forwarding to python process (PID=$!)" && kill -SIGUSR1 $!' USR1

# wait for the python process to finish
wait
