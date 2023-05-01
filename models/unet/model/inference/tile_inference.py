from torch import nn

from model.inference.SingleTilePredictor import SingleTilePredictor


def run_inference(pipeline_config, unet: nn.Module, model_file_name: str = 'unet'):
    # create a mask for the s2_date
    s2_dates = pipeline_config["inference"]["s2_dates"]

    for s2_date in s2_dates:
        single_tile_predictor = SingleTilePredictor(pipeline_config, unet, s2_date, model_file_name)

        single_tile_predictor.infer()
        single_tile_predictor.report_time()
