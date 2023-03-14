echo "All further logs are saved in $LOG_DIR/python.log"

# run the actual task
python main.py --retrain 2>&1 | tee "$LOG_DIR/python.log"
