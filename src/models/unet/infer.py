import os
from collections import OrderedDict

import torch

from model.Model import UNet
from model.inference.tile_inference import run_inference
from src.models.unet.configs.config import BASE_OUTPUT, USE_PIXED_PRECISION, LOAD_CORRUPT_WEIGHTS, DEVICE, report_config
from src.python_helpers.pipeline_config import load_pipeline_config


def main():
    pipeline_config = load_pipeline_config()

    assert pipeline_config['satellite'] == 'sentinel2', "Only Sentinel-2 is supported for inference creation."

    if pipeline_config["inference"]["enable_inference"] != 1:
        print("[INFO] inference is disabled. Skipping...")
        return

    model_file_name = pipeline_config["inference"]["model_file_name"]
    if model_file_name is None or model_file_name == "":
        model_file_name = "unet.pth"

    print(f"[INFO] loading model weights from {model_file_name}...")

    if os.environ.get("RUNS_ON_EULER", 0) == 1:
        model_path = os.path.join(os.environ['MODEL_SAVE_DIR'], model_file_name)
    else:
        model_path = os.path.join(BASE_OUTPUT, model_file_name)

    # run inference

    assert USE_PIXED_PRECISION is False, "Inference with mixed precision is not supported yet."

    unet = UNet().to(DEVICE)

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

    run_inference(pipeline_config, unet, model_file_name=model_file_name)


if __name__ == "__main__":
    report_config()
    main()
