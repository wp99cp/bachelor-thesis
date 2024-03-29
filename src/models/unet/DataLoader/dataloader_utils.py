from imutils import paths
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms as tsfm

from src.models.unet.DataLoader.SegmentationDiskDataset import SegmentationDiskDataset
from src.models.unet.augmentation.Augmentation import Augmentation
from src.models.unet.augmentation.ChannelDropout import ChannelDropout
from src.models.unet.augmentation.HorizontalFlip import HorizontalFlip
from src.models.unet.augmentation.RandomErasing import RandomErasing
from src.models.unet.augmentation.VerticalFlip import VerticalFlip
from src.models.unet.configs.config import IMAGE_DATASET_PATH, MASK_DATASET_PATH, LIMIT_DATASET_SIZE, IMAGE_FLIP_PROB, \
    CHANNEL_DROPOUT_PROB, COVERED_PATCH_SIZE_MIN, COVERED_PATCH_SIZE_MAX, TEST_SPLIT, ENABLE_DATA_AUGMENTATION, \
    PIN_MEMORY, BATCH_SIZE, NUM_DATA_LOADER_WORKERS, BATCH_PREFETCHING


def load_data():
    # load the image and mask filepaths in a sorted manner
    image_paths = sorted(list(paths.list_files(IMAGE_DATASET_PATH, validExts=("npy",))))
    mask_paths = sorted(list(paths.list_files(MASK_DATASET_PATH, validExts=("npy",))))

    if LIMIT_DATASET_SIZE > 0:
        image_paths = image_paths[:LIMIT_DATASET_SIZE]
        mask_paths = mask_paths[:LIMIT_DATASET_SIZE]

    print(f"[INFO] found a total of {len(image_paths)} images in '{IMAGE_DATASET_PATH}'.")
    print(f"[INFO] found a total of {len(mask_paths)} masks in '{MASK_DATASET_PATH}'.")
    assert len(image_paths) == len(mask_paths), "Number of images and masks must match."

    # define transformations
    transforms = tsfm.Compose([tsfm.ToTensor()])
    augmentations: list[Augmentation] = [
        HorizontalFlip(prob=IMAGE_FLIP_PROB),
        VerticalFlip(prob=IMAGE_FLIP_PROB),
        ChannelDropout(prob=CHANNEL_DROPOUT_PROB),
        RandomErasing(prob=CHANNEL_DROPOUT_PROB, min_size=COVERED_PATCH_SIZE_MIN, max_size=COVERED_PATCH_SIZE_MAX)
    ]

    # partition the data into training and testing splits using 85% of
    # the data for training and the remaining 15% for testing
    split = train_test_split(image_paths, mask_paths, test_size=TEST_SPLIT, random_state=42)

    # unpack the data split
    (trainImages, testImages) = split[:2]
    (trainMasks, testMasks) = split[2:]

    print(f"[INFO] loading the dataset from disk. (LIVE_DATASET=0)")

    train_ds = SegmentationDiskDataset(image_paths=trainImages, mask_paths=trainMasks, transforms=transforms,
                                       apply_augmentations=ENABLE_DATA_AUGMENTATION, augmentations=augmentations)
    test_ds = SegmentationDiskDataset(image_paths=testImages, mask_paths=testMasks,
                                      transforms=transforms, apply_augmentations=False)

    # create the training and test data loaders
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE, prefetch_factor=BATCH_PREFETCHING,
                              pin_memory=PIN_MEMORY, num_workers=NUM_DATA_LOADER_WORKERS)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=BATCH_SIZE, prefetch_factor=BATCH_PREFETCHING,
                             pin_memory=PIN_MEMORY, num_workers=NUM_DATA_LOADER_WORKERS)

    print(f"[INFO] loaded {len(train_ds)} examples in the train set.")
    print(f"[INFO] loaded {len(test_ds)} examples in the test set.")
    print(f"\n")

    return train_loader, test_loader, train_ds, test_ds, trainImages, testImages
