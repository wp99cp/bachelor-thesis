from src.models.unet.configs.config import report_config
from src.models.unet.model.testing.testing import run_testing
from src.python_helpers.pipeline_config import load_pipeline_config


def main():
    pipeline_config = load_pipeline_config()

    assert pipeline_config['satellite'] == 'sentinel2', "Only Sentinel-2 is supported for testing."

    if pipeline_config["testing"]["enable"] != 1:
        print("[INFO] testing is disabled. Skipping...")
        return

    model_name = pipeline_config["testing"]["model_name"]

    run_testing(pipeline_config, model_name)


if __name__ == "__main__":
    report_config()
    main()
