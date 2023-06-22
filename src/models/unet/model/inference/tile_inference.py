import os

from torch import nn

from src.models.unet.model.inference.SingleTilePredictor import SingleTilePredictor
from src.python_helpers.pipeline_config import get_dates


def run_inference(pipeline_config, unet: nn.Module, model_file_name: str = 'unet'):
    dates = get_dates(pipeline_config)

    # if empty, we run inference for all dates of the specified tile
    if len(dates) == 0:
        tile_name = pipeline_config["tile_name"]

        base_path = os.environ['EXTRACTED_RAW_DATA']
        folders = os.listdir(base_path)

        # filter for folders that contain the tile name
        folders = [f for f in folders if f'T{tile_name}' in f]
        s2_dates = [f.split('_')[2] for f in folders]
        print(f"[INFO] running inference for all dates of tile {tile_name}: {s2_dates}")
        print(f"[INFO] This is for {len(s2_dates)} dates")
        print("\n")

    tile_id = pipeline_config["tile_id"]
    for date in dates:
        try:
            single_tile_predictor = SingleTilePredictor(pipeline_config, unet, date, tile_id, model_file_name)
            single_tile_predictor.open_scene()
            single_tile_predictor.infer()
            single_tile_predictor.close_scene()
            single_tile_predictor.report_time()

        except AssertionError as e:
            print(f"[ERROR] {e}")
            print(f"[ERROR] Abort during opening scene for {date}")
            continue

        except Exception as e:
            print(f"[ERROR] {e}")
            print(f"[ERROR] Abort during inference for {date}")
            continue
