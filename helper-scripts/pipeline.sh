echo ""
echo "==================="
echo "End-To-End Pipeline"
echo "==================="
echo ""

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

# Load the config yml file
CONFIG_FILE_PATH="$BASE_DIR/$1"
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

if [[ "${config_data_handling_force_rsync}" -eq 0 ]]; then
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

if [[ "${config_data_handling_force_rsync}" -eq 0 ]]; then
  echo "Force rsync is not set. Skip Zip extraction."
else
  echo "Force rsync is set. Extract the zip files."

  find $TMPDIR -type f ! -name '.gitignore' -delete
  find $TMPDIR -type d -empty -delete

  unzip -q "$DATA_RAW_DIR/32TNS_auxiliary_data.zip" -d "$TMPDIR"
  unzip -q "$DATA_RAW_DIR/ExoLabs_classification_S2.zip" -d "$TMPDIR/ExoLabs_classification_S2"

  # extract all zip files in the data/raw folder
  for f in "$DATA_RAW_DIR"/**/*.zip; do
    echo "Extracting $f ..."
    unzip -q "$f" -d "$TMPDIR" &
  done

  wait # wait for all unzip processes to finish

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
  find $ANNOTATED_MASKS_DIR -type f ! -name '.gitignore' -delete
  find $ANNOTATED_MASKS_DIR -type d -empty -delete

  python pre-processing/automatic_masks/automated_masks.py --config_file "$CONFIG_FILE_PATH"

fi

echo ""
echo "==================="
echo "Automated Mask Creation Finished!"
echo "==================="
echo ""

