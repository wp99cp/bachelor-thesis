echo "All further logs are saved in $LOG_DIR"

echo "Hello World!" >"$LOG_DIR/hello_world.txt"

# Log the python version
# shellcheck disable=SC2129
echo "Python version:" >>"$LOG_DIR/hello_world.txt"
python --version >>"$LOG_DIR/hello_world.txt"

# Launch the python script and log the output
echo "Launching the python script" >>"$LOG_DIR/hello_world.txt"
python "$TASK_DIR/python_demo.py" >>"$LOG_DIR/hello_world.txt"
