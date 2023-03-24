echo ""
echo "==================="
echo "End-To-End Pipeline"
echo "==================="
echo ""

# check if environment variable RUNS_ON_EULER is set and "${config_clear_results}" -eq 1
if [[ -z "${RUNS_ON_EULER}" && "${config_clear_results}" -eq 1 ]]; then
  echo "Clear the results directory"
  find "$RESULTS_DIR" -type f ! -name '.gitignore' -delete
  find "$RESULTS_DIR" -type d -empty -delete
fi

# =========================
# Load the config file
# =========================

# See https://stackoverflow.com/a/21189044/13371311
function parse_yaml {
  local prefix=$2
  local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @ | tr @ '\034')
  sed -ne "s|^\($s\):|\1|" \
    -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
    -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p" $1 |
    awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}

CONFIG_FILE_PATH="$BASE_DIR/pipeline-config.yml"

# Load the config yml file
echo "Load Config from '$CONFIG_FILE_PATH'"
parse_yaml "$CONFIG_FILE_PATH" "config_"
echo ""
eval "$(parse_yaml "$CONFIG_FILE_PATH" "config_")"

echo ""
echo "==================="
echo "Loaded!"
echo "==================="
echo ""

# =========================
# Download the data if force download is set to 1
# =========================

if [[ "${config_data_handling_force_rsync}" -eq 0 ]] || [[ -n "${RUNS_ON_EULER}" ]]; then
  echo "Force rsync is not set. Skip rsync of the data."
else
  echo "Force rsync is set. Sync the data."

  # copy the following files and folders recursively from
  # $RAW_DATA_REMOTE_DIR to $DATA_RAW_DIR
  files=('swiss_map_tile.tif' '32TNS_auxiliary_data.zip' 'ExoLabs_classification_S2.zip')
  for f in "${files[@]}"; do
    echo "Copy $f ..."
    rsync -avz --progress "$RAW_DATA_REMOTE_DIR/$f" "$DATA_RAW_DIR"
  done

  if [ ! -d "$DATA_RAW_DIR/raw_data_32TNS_1C" ]; then
    mkdir "$DATA_RAW_DIR/raw_data_32TNS_1C"
  fi
  if [ ! -d "$DATA_RAW_DIR/raw_data_32TNS_2A" ]; then
    mkdir "$DATA_RAW_DIR/raw_data_32TNS_2A"
  fi

  # interpret the string of $config_data_handling_s2_dates as an array
  # it uses the following syntax: ['20210106T102411', '20210406T102021']
  dates_as_array=$(echo "${config_data_handling_s2_dates[@]}" | tr -d '[]')
  IFS=',' read -ra dates_as_array <<<"$dates_as_array" # split the string using ',' as separator

  # remove leading and trailing single quotes
  for i in "${!dates_as_array[@]}"; do
    dates_as_array[$i]=$(echo "${dates_as_array[$i]}" | tr -d "'")
    dates_as_array[$i]=$(echo "${dates_as_array[$i]}" | tr -d " ")
  done

  for d in "${dates_as_array[@]}"; do

    # search for the file that contains the data
    # inside the raw_data_32TNS_1C folder
    file_name=$(find "$RAW_DATA_REMOTE_DIR/raw_data_32TNS_1C" -name "*$d*")
    echo "Found file for $d: $file_name"
    rsync -az --progress "$file_name" "$DATA_RAW_DIR/raw_data_32TNS_1C/" &

    # and search inside the raw_data_32TNS_2A folder
    file_name=$(find "$RAW_DATA_REMOTE_DIR/raw_data_32TNS_2A" -name "*$d*")
    echo "Found file for $d: $file_name"
    rsync -az --progress "$file_name" "$DATA_RAW_DIR/raw_data_32TNS_2A/" &

  done

  wait # wait for all rsync processes to finish

fi

echo ""
echo "==================="
echo "Download Finished!"
echo "==================="
echo ""

# =========================
# Extract Zip files
# =========================

if [[ "${config_data_handling_force_extraction}" -eq 0 ]] && [[ -z "${RUNS_ON_EULER}" ]]; then
  echo "Force rsync is not set. Skip Zip extraction."
else
  echo "Force rsync is set. Extract the zip files."

  if [[ -z "${RUNS_ON_EULER}" ]]; then
    find "$TMP_DIR" -type f ! -name '.gitignore' -delete
    find "$TMP_DIR" -type d -empty -delete
  fi

  ls "$DATA_RAW_DIR"
  unzip -q "$DATA_RAW_DIR/32TNS_auxiliary_data.zip" -d "$TMP_DIR"
  unzip -q "$DATA_RAW_DIR/ExoLabs_classification_S2.zip" -d "$TMP_DIR/ExoLabs_classification_S2"

  # Allow for max 4 concurrent extracting jobs
  MAX_JOBS=4
  active_jobs=0

  # Loop over all zip files in DATA_RAW_DIR and extract them in the background
  for f in "$DATA_RAW_DIR"/**/*.zip; do

    # Wait until the number of active jobs is less than MAX_JOBS
    while ((active_jobs >= MAX_JOBS)); do
      sleep 1
      active_jobs=$(jobs -p | wc -l)
    done

    # Extract zip file in the background and increment active_jobs counter
    echo "Extracting $f ..."
    unzip -q "$f" -d "$TMP_DIR" &
    ((active_jobs++))

  done

  # Wait for all background jobs to complete
  wait

fi

echo ""
echo "==================="
echo "Extracting zip files finished!"
echo "==================="
echo ""

# =========================
# Create automated Masks
# =========================

if [[ "${config_annotation_auto_annotation}" -eq 0 ]]; then
  echo "Skip automated mask creation."
else

  # check if conda is installed
  if command -v conda &>/dev/null; then
    conda activate bachelor_thesis
  else
    echo "conda could not be found, assume all dependencies are installed"
  fi

  echo "Create automated masks"
  echo ""
  find "$ANNOTATED_MASKS_DIR" -type f ! -name '.gitignore' -delete
  find "$ANNOTATED_MASKS_DIR" -type d -empty -delete

  if [[ -z "${RUNS_ON_EULER}" ]]; then
    python3 "$BASE_DIR/pre-processing/automatic_masks/automated_masks.py" --config_file "$CONFIG_FILE_PATH"
  else
    echo "RUN python '$BASE_DIR/pre-processing/automatic_masks/automated_masks.py --config_file $CONFIG_FILE_PATH'"
    python3 -u "$BASE_DIR/pre-processing/automatic_masks/automated_masks.py" --config_file "$CONFIG_FILE_PATH" \
      >"$LOG_DIR/python_automated_masks.log" 2>&1
  fi

fi

echo ""
echo "==================="
echo "Automated Mask Creation Finished!"
echo "==================="
echo ""

# =========================
# Create the training data
# =========================

if [[ "${config_dataset_recreate_dataset}" -eq 0 ]]; then
  echo "Skip automated mask creation."
else

  # check if conda is installed
  if command -v conda &>/dev/null; then
    conda activate bachelor_thesis
  else
    echo "conda could not be found, assume all dependencies are installed"
  fi

  echo "Create the training data"
  echo ""

  find "$DATASET_DIR" -type f ! -name '.gitignore' -delete
  find "$DATASET_DIR" -type d -empty -delete

  if [[ -z "${RUNS_ON_EULER}" ]]; then
    python3 "$BASE_DIR/pre-processing/image_splitter/data-sampler.py" "/annotated_masks"
  else
    python3 -u "$BASE_DIR/pre-processing/image_splitter/data-sampler.py" "/annotated_masks" \
      >"$LOG_DIR/python_data_sampler.log" 2>&1
  fi

fi

echo ""
echo "==================="
echo "Training Data Creation Finished!"
echo "==================="
echo ""

# =========================
# Train Model
# =========================

# check if conda is installed
if command -v conda &>/dev/null; then
  conda activate bachelor_thesis
else
  echo "conda could not be found, assume all dependencies are installed"
fi

if [[ -z "${RUNS_ON_EULER}" ]]; then
  python3 "$BASE_DIR/models/unet/main.py" --retrain
else
  python3 -u "$BASE_DIR/models/unet/main.py" --retrain \
    >"$LOG_DIR/python_model.log" 2>&1
fi
