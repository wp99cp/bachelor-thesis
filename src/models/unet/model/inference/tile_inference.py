import os

from torch import nn

from src.models.unet.model.inference.SingleTilePredictor import SingleTilePredictor
from src.python_helpers.pipeline_config import get_dates


def run_inference(pipeline_config, unet: nn.Module, model_file_name: str = 'unet'):
    dates = get_dates(pipeline_config)

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
