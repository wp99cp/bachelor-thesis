from DataLoader.dataloader_utils import load_data
from configs.config import report_config
from src.python_helpers.pipeline_config import load_pipeline_config
from training import train_unet


def main():
    pipeline_config = load_pipeline_config()
    assert pipeline_config['satellite'] == 'sentinel2', "Only Sentinel-2 is supported for training."

    enable = pipeline_config['training']['enable']
    if not enable:
        print("[INFO] training is disabled. Skipping...")
        return

    train_loader, test_loader, train_ds, test_ds, _, test_images = load_data()

    print("[INFO] retraining model...")
    train_unet(train_loader, test_loader, train_ds, test_ds)


if __name__ == "__main__":
    report_config()
    main()
