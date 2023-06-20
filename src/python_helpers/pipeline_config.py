import os

import yaml


def load_pipeline_config():
    print("Loading the config file...")

    # get the pyth to the config file from the config_file arg
    config_file = os.environ['CONFIG_FILE_PATH']
    print(config_file)

    # check if file exists
    if not os.path.isfile(config_file):
        raise Exception("Config file does not exist.")

    config = None

    with open(config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if config is None:
        print("Could not load the config file.")
        raise Exception("Could not load the config file.")

    return config


def get_dates(pipeline_config):
    print(pipeline_config)
    dates = pipeline_config['dates']

    if len(dates) == 0:
        print("No dates specified in the config file.")

        dates = []
        raw_data_dir = os.path.join(
            os.environ['DATA_DIR'],
            pipeline_config['satellite'],
            "raw_data",
            f"T{pipeline_config['tile_id']}")

        print("Looking for dates in the following directory: ")
        print(raw_data_dir)
        for file in os.listdir(raw_data_dir):

            # skip .gitignore and other files / directories
            if "MSIL1C" not in file:
                continue

            date = file.split("_")[2]
            dates.append(date)

        print("\nFound the following dates: ")
        print(dates)

    return dates
