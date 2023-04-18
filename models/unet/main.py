import os
import sys
from collections import OrderedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.random import multivariate_normal

from DataLoader.dataloader import load_data
from configs.config import report_config, IMAGE_DATASET_PATH, DEVICE, BASE_OUTPUT, LOAD_CORRUPT_WEIGHTS
from model.Model import UNet
from model.inference.patch_inference import make_predictions
from model.inference.tile_inference import tile_inference
from training import train_unet

# import the necessary packages form the pre-processing/image_splitter
sys.path.insert(0, os.environ['BASE_DIR'] + '/helper-scripts/python_helpers')
# noinspection PyUnresolvedReferences
from pipeline_config import load_pipeline_config, get_dates


def print_data_sample(train_loader):
    # show a sample image and mask
    sample = next(iter(train_loader))

    img = sample[0][0].permute(1, 2, 0)
    masks = sample[1][0].permute(1, 2, 0)

    print(f"Image shape: {img.shape}")

    # use matplotlib to display the image and mask
    # show a legend for the mask next to the image
    # i.g. blue = snow, green = clouds, red = water
    matplotlib.use('Agg')
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img[:, :, 0:3])
    ax[0].set_title("Image")

    ax[1].imshow(masks[:, :, 0])
    ax[1].set_title("Mask")

    sample_data_path = os.path.join(BASE_OUTPUT, "sample_data.png")
    plt.savefig(sample_data_path)


def main():
    report_config()

    # check if the flag "--retrain" is set
    # if so, train the model
    emergency_stop = False

    if "--retrain" in sys.argv:
        train_loader, test_loader, train_ds, test_ds, _, test_images = load_data()
        print_data_sample(train_loader)

        print("[INFO] retraining model...")
        emergency_stop = train_unet(train_loader, test_loader, train_ds, test_ds)

        # load the image paths in our testing file and randomly select 10
        # image paths
        print("[INFO] loading up test image paths...")

    # load our model from disk and flash it to the current device
    print("[INFO] load up model...")

    # load the model from disk
    if emergency_stop:
        print("[INFO] emergency stop: skipping model inference.")
        return

    unet = UNet().to(DEVICE)

    pipeline_config = load_pipeline_config()

    model_file_name = pipeline_config["inference"]["model_file_name"]
    if model_file_name is None or model_file_name == "" or os.environ.get("RUNS_ON_EULER", 0) == 1:
        model_file_name = "unet.pth"
    print(f"[INFO] loading model weights from {model_file_name}...")
    model_path = os.path.join(BASE_OUTPUT, model_file_name)

    if not LOAD_CORRUPT_WEIGHTS:  # correct way
        state_dict = torch.load(model_path)

    else:
        # Inference with CPU: use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.
        # original saved file with DataParallel
        state_dict_corrupt = torch.load(model_path)

        # create new OrderedDict that does not contain `module.`
        state_dict = OrderedDict()
        for k, v in state_dict_corrupt.items():
            name = k[17:]  # remove '_orig_mod.module.'
            state_dict[name] = v

    unet.load_state_dict(state_dict)
    unet.print_summary()

    if pipeline_config["inference"]["legacy_mode"]:

        # lost filenames inside IMAGE_DATASET_PATH
        file_names = [f for f in os.listdir(IMAGE_DATASET_PATH) if os.path.isfile(os.path.join(IMAGE_DATASET_PATH, f))]
        print(f"Found {len(file_names)} files in {IMAGE_DATASET_PATH}.")

        for i in range(25):
            # choose a random file
            file = np.random.choice(file_names)
            print(f"Using {file} for prediction.")

            # make predictions and visualize the results
            make_predictions(unet, os.path.join(IMAGE_DATASET_PATH, file))

    else:
        tile_inference(pipeline_config, unet, model_file_name=model_file_name)


if __name__ == "__main__":
    main()
