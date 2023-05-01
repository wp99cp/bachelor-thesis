import os
import sys

import numpy as np
import rasterio
import torch
import tqdm
from pytictac import ClassTimer, accumulate_time, ClassContextTimer  # see https://pypi.org/project/pytictac/
from torch import nn

from config import MEAN_MEANs, MEAN_PERCENTILES_30s, MEAN_PERCENTILES_70s
from configs.config import NUM_CLASSES, DEVICE, BASE_OUTPUT
from model.inference.result_printer import print_results
from utils.encoder_decoder import get_encoded_prediction
from utils.rasterio_helpers import get_profile
from utils.smoothing import smooth_kern

# import the necessary packages form the pre-processing/image_splitter
sys.path.insert(0, os.environ['BASE_DIR'] + '/pre-processing/image_splitter')
# noinspection PyUnresolvedReferences
from SentinelDataLoader import SentinelDataLoader


class SingleTilePredictor:
    """
    This class is responsible for the inference of a single tile (i.e. a single date) of the Sentinel-2 data.
    The inference is done in a sliding window fashion with overlapping patches. Those patches are then stitched
    together to form the full tile using linear blending.
    """

    def __init__(self, pipeline_config, model: nn.Module, s2_date: str, model_file_name='unet', timer_enabled=True):

        """
        :param pipeline_config: the pipeline config (the parsed yaml file)
        :param model: the unet model
        :param s2_date: the date of the Sentinel-2 tile
        :param model_file_name: the name of the model file (default: 'unet') used for loading the weights
        :param timer_enabled: whether the timer should be enabled or not (default: True)
        """

        self.pipeline_config = pipeline_config
        self.model = model
        self.model_file_name = model_file_name
        self.s2_date = s2_date

        self.patch_creator = SentinelDataLoader(dates=[self.s2_date], coverage_mode=True,
                                                border_width=0, timer_enabled=timer_enabled)
        self.cct = ClassTimer(objects=[self], names=["SingleTilePredictor"], enabled=timer_enabled)

        smoother = smooth_kern()
        self.__smoother = np.repeat(smoother[np.newaxis, ...], NUM_CLASSES, axis=0)

    @accumulate_time
    def infer(self):

        """
        Runs the inference for the given date.
        """

        path_prefix = self.__create_output_dir()

        # open the date
        with ClassContextTimer(parent_obj=self, block_name=f"open_{self.s2_date}", parent_method_name="infer"):
            self.patch_creator.open_date(self.s2_date)

        self.__save_contextual_data(path_prefix)

        profile = get_profile(self.s2_date)
        predicted_mask_full_tile = np.zeros((NUM_CLASSES, profile["height"], profile["width"]), dtype='float32')

        limit_patches = self.__get_num_patches()

        # run inference for each patch
        pbar = tqdm.tqdm(range(limit_patches))
        for i in pbar:
            self.__infer_patch(i, path_prefix, pbar, predicted_mask_full_tile)

        self.__save_prediction(path_prefix, predicted_mask_full_tile)

        # this is important, otherwise the inference of the next date will be corrupted
        self.patch_creator.close_date(self.s2_date)

    def __get_num_patches(self):
        # calculate the number of patches if limit_patches is not set
        limit_patches = self.pipeline_config["inference"]["limit_patches"]
        if limit_patches == 0:
            limit_patches = self.patch_creator.get_max_number_of_cover_patches(self.s2_date)
            print(f"Number of patches: {limit_patches}")

        return limit_patches

    @accumulate_time
    def __save_contextual_data(self, path_prefix: str):
        # save bands as RGB jp2 images
        if self.pipeline_config["inference"]["save_RGB_jp2_images"]:
            self.__save_bands(path_prefix, name='TCI', selected_bands=[3, 2, 1])
            self.__save_bands(path_prefix, name='FCI', selected_bands=[12, 7, 3])

        self.__save_training_mask(path=path_prefix)

    def __create_output_dir(self):
        model_name = self.model_file_name.split(".")[0]
        path_prefix = os.path.join(f"{self.s2_date}_pred", model_name)
        # create dir in BASE_OUTPUT/s2_date_pred/model_name
        os.makedirs(os.path.join(BASE_OUTPUT, path_prefix), exist_ok=True)
        # empty the content of the dir
        for file in os.listdir(os.path.join(BASE_OUTPUT, path_prefix)):
            os.remove(os.path.join(BASE_OUTPUT, path_prefix, file))
        return path_prefix

    def __save_prediction(self, path_prefix: str, predicted_mask_full_tile):
        # Create the empty JP2 file
        path = os.path.join(BASE_OUTPUT, path_prefix, f"{self.s2_date}_mask_prediction.jp2")
        with rasterio.open(path, 'w', **get_profile(self.s2_date)) as mask_file:
            mask_file.write(get_encoded_prediction(predicted_mask_full_tile), 1)

    @accumulate_time
    def __infer_patch(self, i, path_prefix: str, pbar: tqdm.std.tqdm, predicted_mask_full_tile):
        # choose a random patch
        (x, y), image, mask = self.patch_creator.next(get_coordinates=True)
        w, h = mask.shape

        # prepare image
        img = np.moveaxis(image, 2, 0)
        img = np.expand_dims(img, 0)
        pbar.set_description(f"Using {(x, y)} for prediction.")

        # make predictions and visualize the results, thus we can reuse the mask generation code
        # for that we turn off gradient tracking and switch the model to evaluation mode
        # the latter is very important, otherwise the results are bad
        # https://discuss.pytorch.org/t/why-the-result-is-changed-after-model-eval/111997/3
        self.model.eval()

        with torch.no_grad():
            image_gpu = torch.from_numpy(img).to(DEVICE)
            _, predicted_mask = self.model(image_gpu)
            predicted_mask = predicted_mask.cpu().numpy().squeeze()

        if i % 250 == 0:
            print_results(image, mask, predicted_mask, f"{x}_{y}", path_prefix)

        # update smoother if shape is not correct
        if self.__smoother.shape[1] != w or self.__smoother.shape[2] != h:
            smoother = smooth_kern(h, w)
            self.__smoother = np.repeat(smoother[np.newaxis, ...], NUM_CLASSES, axis=0)

        # save the patch at the corresponding coordinates
        # we use the smoother to smooth out the mask
        predicted_mask_full_tile[:, x:x + w, y:y + h] += predicted_mask * self.__smoother

    def report_time(self):
        print(f"\nTiming report for {self.s2_date}:")
        print(self.patch_creator.report_time())
        print(self.cct.__str__())
        print()

    @accumulate_time
    def __save_bands(self, path: str, name: str = 'TCI', selected_bands: list[int] = None):
        if selected_bands is None:
            selected_bands = [0, 1, 2]

        assert len(selected_bands) == 3, "selected_bands must be a list of 3 elements"

        bands = self.patch_creator.get_bands(self.s2_date)
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
        profile = get_profile(self.s2_date)
        profile["count"] = 3
        profile["dtype"] = np.uint8  # could also use uint8 to save space or float16 to preserve more information
        with rasterio.open(os.path.join(BASE_OUTPUT, path, f"{name}.jp2"), 'w', **profile) as dst:
            dst.write(image)

    def __save_training_mask(self, path: str):
        mask = self.patch_creator.get_mask(self.s2_date)

        profile = get_profile(self.s2_date)
        profile["dtype"] = np.uint8

        with rasterio.open(os.path.join(BASE_OUTPUT, path, f"training_mask.jp2"), 'w', **profile) as dst:
            dst.write(mask, 1)
