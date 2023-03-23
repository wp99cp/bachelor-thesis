import sys

import yaml


def create_masks(dates):
    print(f"Creating masks for {len(dates)} dates.")

    for date in dates:
        print(f"Creating masks for date {date}.")

        # TODO: create mask


def main():
    # get the pyth to the config file from the config_file arg
    config_file = sys.argv[2]
    print(config_file)

    config = None

    with open(config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if config is None:
        print("Could not load the config file.")
        return

    create_masks(dates=config['data_handling']['s2_dates'])


if __name__ == "__main__":
    main()
