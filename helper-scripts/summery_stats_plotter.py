import json
import os
import sys

import numpy as np
from matplotlib import pyplot as plt


# Function to read in a JSON file and return the data as a dictionary
def read_json_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def main():
    # Read in all the JSON files in the "data" directory
    # read base dir from args
    base_dir = sys.argv[1]

    print(f"Reading in data from {base_dir}...")

    data = []
    for filename in os.listdir(base_dir):

        print(f"Reading in {filename}...")

        if filename.endswith('.json'):
            filepath = os.path.join(base_dir, filename)

            json_data = read_json_file(filepath)

            for elem in json_data:
                elem["date"] = filename.split(".")[0]

            data.extend(json_data)

    print(f"Read in {len(data)} measurements.")

    # create an empty list for each band
    bands_means = {}
    bands_maxs = {}
    bands_mins = {}
    percentile_95s = {}
    percentile_70s = {}
    percentile_30s = {}
    percentile_5s = {}

    # iterate over the data
    for elem in data:

        band_name = elem['band']
        if band_name == "ELEV": continue

        if band_name not in bands_means:
            bands_means[band_name] = []
            bands_maxs[band_name] = []
            bands_mins[band_name] = []
            percentile_95s[band_name] = []
            percentile_70s[band_name] = []
            percentile_30s[band_name] = []
            percentile_5s[band_name] = []

        # append the mean value of the band to the list
        bands_means[band_name].append(elem['raw']["mean"])
        bands_maxs[band_name].append(elem['raw']["max"])
        bands_mins[band_name].append(elem['raw']["min"])
        percentile_95s[band_name].append(elem['raw']["percentile_95"])
        percentile_5s[band_name].append(elem['raw']["percentile_5"])
        percentile_70s[band_name].append(elem['raw']["percentile_70"])
        percentile_30s[band_name].append(elem['raw']["percentile_30"])

    print(bands_means.items())

    print(f"min_mean: ")
    for band_name, bands_min in bands_mins.items():
        print(round(np.min(bands_min)), end=", ")
    print()

    print(f"mean_percentile_5: ")
    for band_name, percentile_5 in percentile_5s.items():
        print(round(np.mean(percentile_5)), end=", ")
    print()

    print(f"mean_mean: ")
    for band_name, bands_mean in bands_means.items():
        print(round(np.mean(bands_mean)), end=", ")
    print()

    print(f"mean_percentile_95: ")
    for band_name, percentile_95 in percentile_95s.items():
        print(round(np.mean(percentile_95)), end=", ")
    print()

    print(f"max_max: ")
    for band_name, bands_max in bands_maxs.items():
        print(round(np.max(bands_max)), end=", ")
    print()

    print(f"mean_percentile_70: ")
    for band_name, percentile_70 in percentile_70s.items():
        print(round(np.mean(percentile_70)), end=", ")
    print()

    print(f"mean_percentile_30: ")
    for band_name, percentile_30 in percentile_30s.items():
        print(round(np.mean(percentile_30)), end=", ")
    print()

    # create a boxplot for each band
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))

    axs[0].set_title("Mean")
    axs[0].boxplot(bands_means.values(), labels=bands_means.keys())

    axs[1].set_title("Max")
    axs[1].boxplot(bands_maxs.values(), labels=bands_maxs.keys())

    axs[2].set_title("Min")
    axs[2].boxplot(bands_mins.values(), labels=bands_mins.keys())

    axs[3].set_title("95th Percentile")
    axs[3].boxplot(percentile_95s.values(), labels=percentile_95s.keys())

    axs[4].set_title("5th Percentile")
    axs[4].boxplot(percentile_5s.values(), labels=percentile_5s.keys())

    # save the figure
    plt.savefig("boxplot.png")


if __name__ == "__main__":
    main()
