
echo "All further logs are saved in $LOG_DIR"

python main.py --retrain >> "$LOG_DIR/python.log"