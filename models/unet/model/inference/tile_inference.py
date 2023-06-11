import os

from torch import nn

from model.inference.SingleTilePredictor import SingleTilePredictor


def run_inference(pipeline_config, unet: nn.Module, model_file_name: str = 'unet'):
    # create a mask for the s2_date
    s2_dates = pipeline_config["inference"]["s2_dates"]

    # if empty, we run inference for all dates of the specified tile
    if len(s2_dates) == 0:
        tile_name = pipeline_config["tile_name"]

        base_path = os.environ['EXTRACTED_RAW_DATA']
        folders = os.listdir(base_path)

        # filter for folders that contain the tile name
        folders = [f for f in folders if f'T{tile_name}' in f]
        s2_dates = [f.split('_')[2] for f in folders]
        print(f"[INFO] running inference for all dates of tile {tile_name}: {s2_dates}")

    for s2_date in s2_dates:
        single_tile_predictor = SingleTilePredictor(pipeline_config, unet, s2_date, model_file_name)

        single_tile_predictor.infer()
        single_tile_predictor.report_time()
