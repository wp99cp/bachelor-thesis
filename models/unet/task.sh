echo "All further logs are saved in $LOG_DIR/python.log"

# run the actual task
python main.py --retrain >>"$LOG_DIR/python.log" &

# trap om USR1 and forward the signal to the python process
# this is used to save the model before the slurm job is killed
trap 'kill -USR1 $!' USR1

wait
