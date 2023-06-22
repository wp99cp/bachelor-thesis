import os

import numpy as np
import rasterio
import torch
import tqdm
from pytictac import ClassTimer, accumulate_time, ClassContextTimer  # see https://pypi.org/project/pytictac/
from torch import nn

from src.datahandler.NormalizedDataHandler import NormalizedDataHandler, MEAN_PERCENTILES_70s, MEAN_MEANs, \
    MEAN_PERCENTILES_30s
from src.datahandler.auxiliary_reader.AuxiliaryReader import AuxiliaryData
from src.datahandler.patch_creator.CoveragePatchCreator import CoveragePatchCreator
from src.datahandler.satallite_reader.SentinelL1CReader import SentinelL1CReader, Bands
from src.models.unet.configs.config import NUM_CLASSES, IMAGE_SIZE, DEVICE, BASE_OUTPUT
from src.models.unet.utils.encoder_decoder import get_encoded_prediction
from src.models.unet.utils.smoothing import smooth_kern


class SingleTilePredictor:
    """
    This class is responsible for the inference of a single tile (i.e. a single date) of the Sentinel-2 data.
    The inference is done in a sliding window fashion with overlapping patches. Those patches are then stitched
    together to form the full tile using linear blending.
    """

    def __init__(self, pipeline_config, model: nn.Module, s2_date: str, tile_id: str, model_file_name='unet',
                 timer_enabled=True):

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
        self.date = s2_date
        self.tile_id = tile_id

        satellite_reader = SentinelL1CReader()
        dataloader = NormalizedDataHandler(satellite_reader=satellite_reader)
        self.profile = satellite_reader.get_profile(date=s2_date, resolution=10, tile_id=tile_id)

        self.patch_creator = CoveragePatchCreator(
            dataloader=dataloader,
            patch_size=IMAGE_SIZE,
            auxiliary_data=[AuxiliaryData.DEM],
            bands=None  # none means all bands, but is a lot faster than specifying all bands
        )

        self.cct = ClassTimer(objects=[self], names=["SingleTilePredictor"], enabled=timer_enabled)

        smoother = smooth_kern()
        self.__smoother = np.repeat(smoother[np.newaxis, ...], NUM_CLASSES, axis=0)

    @accumulate_time
    def infer(self):

        """
        Runs the inference for the given date.
        """
        path_prefix = self.__create_output_dir()
        self.__save_contextual_data(path_prefix)

        predicted_mask_full_tile = np.zeros((
            NUM_CLASSES,
            self.profile["height"],
            self.profile["width"]),
            dtype='float32')

        limit_patches = self.__get_num_patches()

        # run inference for each patch
        pbar = tqdm.tqdm(range(limit_patches))
        for _ in pbar:
            self.__infer_patch(pbar, predicted_mask_full_tile)

        self.__save_prediction(path_prefix, predicted_mask_full_tile)

    def __get_num_patches(self):
        # calculate the number of patches if limit_patches is not set
        limit_patches = self.pipeline_config["inference"]["limit_patches"]

        if limit_patches == 0:
            return self.patch_creator.get_patch_count()
        return limit_patches

    @accumulate_time
    def open_scene(self):
        assert self.patch_creator.dataloader is not None, "Dataloader is not initialized"

        assert self.date is not None, "S2 date is not initialized"
        assert self.tile_id is not None, "Tile id is not initialized"

        self.patch_creator.dataloader.open_scene(tile_id=self.tile_id, date=self.date)

    @accumulate_time
    def close_scene(self):
        self.patch_creator.dataloader.close_scene()

    @accumulate_time
    def __save_contextual_data(self, path_prefix: str):
        # save bands as RGB jp2 images
        if self.pipeline_config["inference"]["save_RGB_jp2_images"]:
            self.__save_bands(path_prefix, name='TCI', selected_bands=[Bands.B04, Bands.B03, Bands.B02])
            self.__save_bands(path_prefix, name='FCI', selected_bands=[Bands.B12, Bands.B8A, Bands.B04])

        if self.pipeline_config["inference"]["save_mask"]:
            self.__save_training_mask(path=path_prefix)
        self.__save_data_coverage(path_prefix)

    def __save_data_coverage(self, path_prefix: str):

        data_coverage = self.patch_creator.dataloader.get_satellite_data_coverage()

        # save data coverage as jp2 image
        if self.pipeline_config["inference"]["save_data_coverage"]:
            path = os.path.join(BASE_OUTPUT, path_prefix, f"{self.date}_data_coverage.jp2")
            with rasterio.open(path, 'w', **self.profile) as data_coverage_file:
                data_coverage_file.write(data_coverage, 1)

    def __create_output_dir(self):
        model_name = self.model_file_name.split(".")[0]
        path_prefix = os.path.join(f"{self.tile_id}_{self.date}_pred", model_name)
        # create dir in BASE_OUTPUT/s2_date_pred/model_name
        os.makedirs(os.path.join(BASE_OUTPUT, path_prefix), exist_ok=True)
        # empty the content of the dir
        for file in os.listdir(os.path.join(BASE_OUTPUT, path_prefix)):
            os.remove(os.path.join(BASE_OUTPUT, path_prefix, file))
        return path_prefix

    def __save_prediction(self, path_prefix: str, predicted_mask_full_tile):

        if self.pipeline_config["inference"]["save_raw_predictions"]:
            raw_profile = self.profile.copy()
            raw_profile["dtype"] = 'float32'
            raw_profile["count"] = 4

            path_raw = os.path.join(BASE_OUTPUT, path_prefix, f"{self.date}_mask_prediction_raw.jp2")
            with rasterio.open(path_raw, 'w', **raw_profile) as mask_file_raw:
                mask_file_raw.write(predicted_mask_full_tile[0, :, :], 1)
                mask_file_raw.write(predicted_mask_full_tile[1, :, :], 2)
                mask_file_raw.write(predicted_mask_full_tile[2, :, :], 3)
                mask_file_raw.write(predicted_mask_full_tile[3, :, :], 4)

        # Create the empty JP2 file
        path = os.path.join(BASE_OUTPUT, path_prefix, f"{self.date}_mask_prediction.jp2")
        with rasterio.open(path, 'w', **self.profile) as mask_file:
            mask_file.write(get_encoded_prediction(predicted_mask_full_tile, False), 1)

        if self.pipeline_config["inference"]["save_raw_thresholded"]:
            path = os.path.join(BASE_OUTPUT, path_prefix, f"{self.date}_mask_prediction_thresholded.jp2")
            with rasterio.open(path, 'w', **self.profile) as mask_file:
                mask_file.write(get_encoded_prediction(predicted_mask_full_tile, True), 1)

    @accumulate_time
    def __infer_patch(self, pbar: tqdm.std.tqdm, predicted_mask_full_tile):

        # choose a random patch
        with ClassContextTimer(parent_obj=self, block_name="get_next_patch", parent_method_name="__infer_patch"):
            (x, y), image = self.patch_creator.get_next_patch(tile_id=self.tile_id, date=self.date,
                                                              include_mask=False)

        (w, h) = image.shape[1:]
        pbar.set_description(f"Using {(x, y)} for prediction.")

        # make predictions and visualize the results, thus we can reuse the mask generation code
        # for that we turn off gradient tracking and switch the model to evaluation mode
        # the latter is very important, otherwise the results are bad
        # https://discuss.pytorch.org/t/why-the-result-is-changed-after-model-eval/111997/3
        self.model.eval()

        with ClassContextTimer(parent_obj=self, block_name="infer_single_patch", parent_method_name="__infer_patch"):
            with torch.no_grad():
                image_gpu = torch.from_numpy(image).to(DEVICE)

                if self.pipeline_config["inference"]["use_all_rotations"]:
                    # Create rotated versions of the images
                    images_rotated = torch.flip(image_gpu, [2])
                    images_rotated_flip = torch.flip(image_gpu, [1])
                    images_rotated_flip_both = torch.flip(image_gpu, [1, 2])

                    # Concatenate all the images together
                    all_images = torch.stack(
                        [image_gpu, images_rotated, images_rotated_flip, images_rotated_flip_both]
                    )

                    # Make predictions for all the images at once
                    _, predicted_masks = self.model(all_images)

                    # Split the predicted masks for each rotation
                    predicted_mask_up = predicted_masks[0, :, :, :]
                    predicted_mask_left = predicted_masks[1, :, :, :]
                    predicted_mask_right = predicted_masks[2, :, :, :]
                    predicted_mask_down = predicted_masks[3, :, :, :]

                    # Calculate the mean of the predictions
                    predicted_mask = (predicted_mask_up + torch.flip(predicted_mask_left, [2]) +
                                      torch.flip(predicted_mask_right, [1]) + torch.flip(predicted_mask_down,
                                                                                         [1, 2])) / 4
                else:
                    image_gpu = image_gpu.unsqueeze(0)
                    _, predicted_mask = self.model(image_gpu)
                    predicted_mask = predicted_mask.squeeze(0)

                predicted_mask = predicted_mask.cpu().numpy()

        with ClassContextTimer(parent_obj=self, block_name="apply_smoothing_kernel",
                               parent_method_name="__infer_patch"):
            # update smoother if shape is not correct
            if self.__smoother.shape[1] != w or self.__smoother.shape[2] != h:
                smoother = smooth_kern(h, w)
                self.__smoother = np.repeat(smoother[np.newaxis, ...], NUM_CLASSES, axis=0)

            # save the patch at the corresponding coordinates
            # we use the smoother to smooth out the mask
            predicted_mask_full_tile[:, x:x + w, y:y + h] += predicted_mask * self.__smoother

    def report_time(self):
        print(f"\nTiming report for {self.date}:")
        print(self.patch_creator.report_timing())
        print(self.cct.__str__())
        print()

    @accumulate_time
    def __save_bands(self, path: str, name: str = 'TCI', selected_bands: list[Bands] = None):

        assert len(selected_bands) == 3, "selected_bands must be a list of 3 elements"

        image = self.patch_creator.dataloader.get_bands(bands=selected_bands)

        selected_bands_idx = [b.value[1] for b in selected_bands]

        # shift back by adding mean channel values
        a = -1.7346  # 0.15 == 1 / (1 + Exp[-x]) // Solve
        b = +1.7346  # 0.85 == 1 / (1 + Exp[-x]) // Solve
        c = np.log10(MEAN_PERCENTILES_30s)[selected_bands_idx]
        d = np.log10(MEAN_PERCENTILES_70s)[selected_bands_idx]

        offset = np.log10(MEAN_MEANs)[selected_bands_idx]
        offset = (offset - c) * (b - a) / (d - c) + a
        offset = 1 / (1 + np.exp(-offset))
        np.add(image, offset.reshape((-1, 1, 1)), out=image)

        np.clip(image, 0, 1, out=image)
        np.multiply(image, 255, out=image)
        image = image.astype(np.uint8)

        assert image.shape[0] == 3, "image must have 3 channels"

        # use rasterio to save the tci
        profile = self.profile.copy()
        profile["count"] = 3
        profile["dtype"] = np.uint8  # could also use uint8 to save space or float16 to preserve more information
        with rasterio.open(os.path.join(BASE_OUTPUT, path, f"{name}.jp2"), 'w', **profile) as dst:
            dst.write(image)

    def __save_training_mask(self, path: str):

        mask = self.patch_creator.dataloader.get_masks()

        if mask is None:
            print(f"No training data for {self.date}")
            return  # no training data for this tile

        profile = self.profile.copy()
        profile["dtype"] = np.uint8

        with rasterio.open(os.path.join(BASE_OUTPUT, path, f"training_mask.jp2"), 'w', **profile) as dst:
            dst.write(mask, 1)
