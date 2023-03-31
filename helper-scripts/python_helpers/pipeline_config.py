import os

import yaml

LIMIT_DATES = int(os.environ.get('LIMIT_DATES', 0))


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


EXTRACTED_RAW_DATA = os.environ['EXTRACTED_RAW_DATA']


def get_dates(pipeline_config, pipeline_step="annotation"):
    dates = pipeline_config[pipeline_step]['s2_dates']

    if len(dates) == 0 or LIMIT_DATES > 0:
        print("No dates specified in the config file.")

        dates = []
        for file in os.listdir(EXTRACTED_RAW_DATA):

            # skip .gitignore and other files / directories
            if "MSIL1C" not in file:
                continue

            # print(file)

            date = file.split("_")[2]
            dates.append(date)

        print("Found the following dates: ")
        print(dates)

        if LIMIT_DATES > 0:
            dates = dates[:LIMIT_DATES]
            print("Limiting the dates to the first {} dates.".format(LIMIT_DATES))
            print(dates)

        print("")

    return dates
