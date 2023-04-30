import os
import sys

import numpy as np
import rasterio
import torch
import tqdm

from config import MEAN_MEANs, MEAN_PERCENTILES_30s, MEAN_PERCENTILES_70s
from configs.config import NUM_CLASSES, DEVICE, BASE_OUTPUT
from model.inference.patch_inference import print_results
from utils.encoder_decoder import get_encoded_prediction
from utils.rasterio_helpers import get_profile
from utils.smoothing import smooth_kern

# import the necessary packages form the pre-processing/image_splitter
sys.path.insert(0, os.environ['BASE_DIR'] + '/pre-processing/image_splitter')
# noinspection PyUnresolvedReferences
from SentinelDataLoader import SentinelDataLoader


def tile_inference(pipeline_config, unet, model_file_name='unet'):
    # create a mask for the s2_date
    s2_dates = pipeline_config["inference"]["s2_dates"]

    for s2_date in s2_dates:
        model_name = model_file_name.split(".")[0]
        path_prefix = os.path.join(f"{s2_date}_pred", model_name)

        __tile_inference_of_date(s2_date, pipeline_config, unet, path_prefix)


def __tile_inference_of_date(s2_date, pipeline_config, unet, path_prefix):
    # create dir in BASE_OUTPUT/s2_date_pred/model_name
    os.makedirs(os.path.join(BASE_OUTPUT, path_prefix), exist_ok=True)

    # empty the content of the dir
    for file in os.listdir(os.path.join(BASE_OUTPUT, path_prefix)):
        os.remove(os.path.join(BASE_OUTPUT, path_prefix, file))

    limit_patches = pipeline_config["inference"]["limit_patches"]
    patch_creator = SentinelDataLoader(dates=[s2_date], coverage_mode=True, border_width=0)
    patch_creator.open_date(s2_date)

    # calculate the number of patches if limit_patches is not set
    if limit_patches == 0:
        limit_patches = patch_creator.get_max_number_of_cover_patches(s2_date)
        print(f"Number of patches: {limit_patches}")

    # save bands as RGB jp2 images
    if pipeline_config["inference"]["save_RGB_jp2_images"]:
        _save_bands(patch_creator, s2_date, path=path_prefix, name='TCI', selected_bands=[3, 2, 1])
        _save_bands(patch_creator, s2_date, path=path_prefix, name='FCI', selected_bands=[12, 7, 3])
    _save_training_mask(patch_creator, s2_date, path=path_prefix)

    profile = get_profile(s2_date)
    predicted_mask_full_tile = np.zeros((NUM_CLASSES, profile["height"], profile["width"]), dtype='float32')
    gaussian_smoother = smooth_kern()

    # copy NUM_CLASSES times the gaussian_smoother
    gaussian_smoother = np.repeat(gaussian_smoother[np.newaxis, ...], NUM_CLASSES, axis=0)
    pbar = tqdm.tqdm(range(limit_patches))

    for i in pbar:
        # choose a random patch
        (x, y), image, mask = patch_creator.next(get_coordinates=True)
        w, h = mask.shape

        # prepare image
        img = np.moveaxis(image, 2, 0)
        img = np.expand_dims(img, 0)

        pbar.set_description(f"Using {(x, y)} for prediction.")

        # make predictions and visualize the results, thus we can reuse the mask generation code
        # for that we turn off gradient tracking and switch the model to evaluation mode
        # the latter is very important, otherwise the results are bad
        # https://discuss.pytorch.org/t/why-the-result-is-changed-after-model-eval/111997/3
        unet.eval()
        with torch.no_grad():
            image_gpu = torch.from_numpy(img).to(DEVICE)
            _, predicted_mask = unet(image_gpu)
            predicted_mask = predicted_mask.cpu().numpy().squeeze()

        if i % 250 == 0:
            print_results(image, mask, predicted_mask, f"{x}_{y}", path_prefix)

        # update gaussian_smoother if shape is not correct
        if gaussian_smoother.shape[1] != w or gaussian_smoother.shape[2] != h:
            gaussian_smoother = smooth_kern(h, w)
            gaussian_smoother = np.repeat(gaussian_smoother[np.newaxis, ...], NUM_CLASSES, axis=0)

        # save the patch at the corresponding coordinates
        # we use the gaussian to smooth out the mask
        predicted_mask_full_tile[:, x:x + w, y:y + h] += predicted_mask * gaussian_smoother

    # Create the empty JP2 file
    path = os.path.join(BASE_OUTPUT, path_prefix, f"{s2_date}_mask_prediction.jp2")
    with rasterio.open(path, 'w', **profile) as mask_file:
        mask_file.write(get_encoded_prediction(predicted_mask_full_tile), 1)

    # this is important, otherwise the inference of the next date will be corrupted
    patch_creator.close_date(s2_date)


def _save_bands(patch_creator, s2_date, path, name='TCI', selected_bands=None):
    if selected_bands is None:
        selected_bands = [0, 1, 2]

    assert len(selected_bands) == 3, "selected_bands must be a list of 3 elements"

    bands = patch_creator.get_bands(s2_date)
    # print(f"bands.shape: {bands.shape}")
    image = bands[selected_bands, :, :]

    # shift back by adding mean channel values
    a = -1.7346  # 0.15 == 1 / (1 + Exp[-x]) // Solve
    b = +1.7346  # 0.85 == 1 / (1 + Exp[-x]) // Solve
    c = np.log10(np.array(MEAN_PERCENTILES_30s)[selected_bands])
    d = np.log10(np.array(MEAN_PERCENTILES_70s)[selected_bands])
    offset = np.log10(np.array(MEAN_MEANs)[selected_bands])
    offset = (offset - c) * (b - a) / (d - c) + a
    offset = 1 / (1 + np.exp(-offset))

    image = image + offset[:, np.newaxis, np.newaxis]
    image = np.clip(image, 0, 1)
    image = image * 255.0
    image = image.astype(np.uint8)

    # lift zero values to 0.1
    # to prevent qgis to show those areas as no data (white)
    image[image == 0] = 1
    assert image.shape[0] == 3, "image must have 3 channels"

    # use rasterio to save the tci
    profile = get_profile(s2_date)
    profile["count"] = 3
    profile["dtype"] = np.uint8  # could also use uint8 to save space or float16 to preserve more information
    with rasterio.open(os.path.join(BASE_OUTPUT, path, f"{name}.jp2"), 'w', **profile) as dst:
        dst.write(image)


def _save_training_mask(patch_creator, s2_date, path):
    mask = patch_creator.get_mask(s2_date)

    profile = get_profile(s2_date)
    profile["dtype"] = np.uint8

    with rasterio.open(os.path.join(BASE_OUTPUT, path, f"training_mask.jp2"), 'w', **profile) as dst:
        dst.write(mask, 1)
