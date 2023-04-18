import os
import sys

import numpy as np
import rasterio
import torch
import tqdm

from configs.config import IMAGE_SIZE, THRESHOLDED_PREDICTION, THRESHOLD, NUM_CLASSES, DEVICE, BASE_OUTPUT
from model.inference.patch_inference import print_results

# import the necessary packages form the pre-processing/image_splitter
sys.path.insert(0, os.environ['BASE_DIR'] + '/pre-processing/image_splitter')
# noinspection PyUnresolvedReferences
from RandomPatchCreator import RandomPatchCreator


def _get_base_path(date):
    EXTRACTED_RAW_DATA = os.environ['EXTRACTED_RAW_DATA']

    folders = os.listdir(EXTRACTED_RAW_DATA)
    folders = [f for f in folders if f"_MSIL1C_{date}" in f]
    folder = folders[0]

    base_path = f"{EXTRACTED_RAW_DATA}/{folder}/GRANULE/"
    sub_folder = os.listdir(base_path)
    base_path += sub_folder[0] + '/IMG_DATA'

    return base_path


def __smooth_kern(kernel_size=IMAGE_SIZE):
    """

    """

    heaviside_lambda = lambda x: -np.sign(x) * x + 1
    heaviside_lambda_2D = lambda x, y: heaviside_lambda(x) * heaviside_lambda(y)

    xs = np.linspace(-1, 1, kernel_size)
    ys = np.linspace(-1, 1, kernel_size)
    xv, yv = np.meshgrid(xs, ys)

    return heaviside_lambda_2D(xv, yv)


def _get_profile(s2_date):
    base_path = _get_base_path(s2_date)
    B02 = f"T32TNS_{s2_date}_B02.jp2"
    B02 = rasterio.open(f"{base_path}/{B02}")

    profile = {
        'driver': 'GTiff',
        'dtype': np.uint8,
        'nodata': 0,
        'width': B02.width,
        'height': B02.height,
        'count': 1,
        'crs': B02.crs,
        'transform': B02.transform,
        'blockxsize': 512,
        'blockysize': 512,
        'compress': 'lzw',
    }

    return profile


def _get_encoded_prediction(predicted_mask):
    print(f"Min/Max of predicted_mask: {np.min(predicted_mask)}/{np.max(predicted_mask)}")
    if THRESHOLDED_PREDICTION:
        predicted_mask[:, :, :] = (predicted_mask[:, :, :] > THRESHOLD).astype(int)
    return np.argmax(predicted_mask, axis=0)


def tile_inference(pipeline_config, unet, model_file_name='unet'):
    # create a mask for the s2_date
    s2_date = pipeline_config["inference"]["s2_date"]

    model_name = model_file_name.split(".")[0]
    path_prefix = os.path.join(f"{s2_date}_pred", model_name)

    # create dir in BASE_OUTPUT/s2_date_pred/model_name
    os.makedirs(os.path.join(BASE_OUTPUT, path_prefix), exist_ok=True)

    # empty the content of the dir
    for file in os.listdir(os.path.join(BASE_OUTPUT, path_prefix)):
        os.remove(os.path.join(BASE_OUTPUT, path_prefix, file))

    limit_patches = pipeline_config["inference"]["limit_patches"]
    assert (limit_patches > 0, "limit_patches must be greater than 0")
    patch_creator = RandomPatchCreator(dates=[s2_date], coverage_mode=True)
    patch_creator.open_date(s2_date)

    # save bands as RGB jp2 images
    _save_bands(patch_creator, s2_date, path=path_prefix, name='TCI', selected_bands=[3, 2, 1])
    _save_bands(patch_creator, s2_date, path=path_prefix, name='FCI', selected_bands=[12, 7, 3])

    profile = _get_profile(s2_date)
    predicted_mask_full_tile = np.zeros((NUM_CLASSES, profile["height"], profile["width"]), dtype='float32')
    gaussian_smoother = __smooth_kern(IMAGE_SIZE)

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

        # save the patch at the corresponding coordinates
        # we use the gaussian to smooth out the mask
        predicted_mask_full_tile[:, x:x + w, y:y + h] += predicted_mask * gaussian_smoother

    # Create the empty JP2 file
    path = os.path.join(BASE_OUTPUT, path_prefix, f"{s2_date}_mask_prediction.jp2")
    with rasterio.open(path, 'w', **profile) as mask_file:
        mask_file.write(_get_encoded_prediction(predicted_mask_full_tile), 1)


def _save_bands(patch_creator, s2_date, path, name='TCI', selected_bands=None):
    if selected_bands is None:
        selected_bands = [0, 1, 2]

    assert len(selected_bands) == 3, "selected_bands must be a list of 3 elements"

    bands = patch_creator.get_bands(s2_date)
    print(f"bands.shape: {bands.shape}")
    image = bands[selected_bands, :, :]

    # use rasterio to save the tci
    profile = _get_profile(s2_date)
    profile["count"] = 3
    profile["dtype"] = np.uint8
    with rasterio.open(os.path.join(BASE_OUTPUT, path, f"{name}.jp2"), 'w', **profile) as dst:
        dst.write(image)
