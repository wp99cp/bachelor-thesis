echo "All further logs are saved in $LOG_DIR/python.log"

# Define a signal handler function to forward USR1 signals to the Python process
handle_signal() {
    if [ $1 = "USR1" ]; then
        echo "Forwarding USR1 signal to Python process"
        kill -USR1 "$PYTHON_PID"
    fi
}

# Set the signal handler function for USR1 signals
trap 'handle_signal USR1' USR1

# Start the Python process in the background and save its PID
python main.py retrain 3>&1 1>>$LOG_DIR/python.log 2>&1 &
PYTHON_PID=$!

# Wait for the Python process to finish
echo "Waiting for Python process (PID=$PYTHON_PID) to finish"
wait $PYTHON_PID
