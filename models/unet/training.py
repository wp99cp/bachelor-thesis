import json
import os

import torch
from matplotlib import pyplot as plt
from torch.nn import BCEWithLogitsLoss, DataParallel
from torch.optim import RMSprop, lr_scheduler, Adam, AdamW

from configs.config import DEVICE, INIT_LR, BASE_OUTPUT, IMAGE_SIZE, CLASS_WEIGHTS, MOMENTUM, \
    WEIGHT_DECAY, EARLY_STOPPING_PATIENCE, WEIGHT_DECAY_PLATEAU_PATIENCE, CONTINUE_TRAINING
from model.EarlyStopping import EarlyStopping
from model.Model import UNet
from model.ModelTrainer import ModelTrainer
from model.metrices import get_segmentation_metrics


def train_unet(train_loader, test_loader, train_ds, test_ds):
    # initialize our UNet model
    unet = UNet()

    # print model summary
    unet.to(DEVICE).print_summary(3, step_up=True, show_hierarchical=True)

    # if continue training, load the model
    if CONTINUE_TRAINING:
        print("[INFO] Training will continue from previous checkpoint...")
        print("[INFO] loading the weights...")
        model_path = os.path.join(BASE_OUTPUT, "unet.pth")
        unet.load_state_dict(torch.load(model_path))

    unet = DataParallel(unet)  # allow multiple GPUs
    unet = unet.to(DEVICE)  # move the model to the GPU

    # if torch version 2.0 or higher, we compile the model
    if torch.__version__ >= "2.0":
        print(f"[INFO] compiling the model with torch {torch.__version__}...")

        # check if cuda compatibility is greater or equal to 7.0
        if torch.cuda.get_device_capability()[0] >= 7:
            unet = torch.compile(unet)
        else:
            print("[INFO] cuda version is too old, skipping compilation...")

    # the classes are unbalanced, so we need to artificially increase the
    # weight of the positive classes
    class_weights = get_class_weights()

    # initialize loss function and optimizer
    loss_func = BCEWithLogitsLoss(pos_weight=class_weights.to(DEVICE), reduction='mean')
    opt = AdamW(unet.parameters(), lr=INIT_LR, amsgrad=True, betas=(0.9, 0.999), eps=1e-6)
    # opt = Adam(unet.parameters(), lr=INIT_LR, amsgrad=True, betas=(0.9, 0.999))
    # opt = RMSprop(unet.parameters(), lr=INIT_LR, momentum=MOMENTUM)
    scheduler = lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=WEIGHT_DECAY_PLATEAU_PATIENCE,
                                               factor=WEIGHT_DECAY, verbose=True)
    es = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, min_delta=0, restore_best_weights=True)
    metrics = get_segmentation_metrics()

    trainer = ModelTrainer(unet, loss_func, opt, scheduler, es, metrics)

    print("[INFO] training the network...")
    trainer.train(train_ds=train_ds, test_ds=test_ds, train_loader=train_loader, test_loader=test_loader)

    print(trainer.get_timing_summary())

    # save history as JSON
    print("[INFO] saving the training history...")

    history = trainer.get_history()

    save_history(history)

    # save the model
    print("[INFO] saving the model...")
    model_name = "unet.pth" if not trainer.emergency_stop else "unet_emergency.pth"
    model_path = os.path.join(BASE_OUTPUT, model_name)
    torch.save(unet.module.state_dict(), model_path)

    plot_history(history)

    print("[INFO] Training completed!\n")
    return trainer.emergency_stop


def plot_history(history):
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 10))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["test_loss"], label="test_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend(loc="lower left")
    loss_plot_path = os.path.join(BASE_OUTPUT, "loss.png")
    plt.savefig(loss_plot_path)


def save_history(history):
    for key, value in history.items():
        history[key] = [v.tolist() for v in value]
    history_file = open(os.path.join(BASE_OUTPUT, "history.json"), "w")
    history_file.write(json.dumps(history))
    history_file.close()


def get_class_weights():
    class_weights = torch.tensor(CLASS_WEIGHTS)

    # see https://stackoverflow.com/a/69832861/13371311
    class_weights = (1 - class_weights) / class_weights
    print("Class weights: ", class_weights)

    # repeat the weights for each pixel in the image
    class_weights = class_weights.repeat(IMAGE_SIZE, IMAGE_SIZE, 1)
    class_weights = class_weights.permute(2, 0, 1)
    return class_weights
