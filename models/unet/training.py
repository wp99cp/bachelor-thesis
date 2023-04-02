import json
import os

import torch
from matplotlib import pyplot as plt
from torch.nn import BCEWithLogitsLoss
from torch.optim import RMSprop, lr_scheduler

from configs.config import DEVICE, INIT_LR, BASE_OUTPUT, IMAGE_SIZE, CLASS_WEIGHTS, MOMENTUM, \
    WEIGHT_DECAY
from model.EarlyStopping import EarlyStopping
from model.Model import UNet
from model.ModelTrainer import ModelTrainer
from model.metrices import get_segmentation_metrics


def train_unet(train_loader, test_loader, train_ds, test_ds):
    # initialize our UNet model
    unet = UNet().to(DEVICE)
    unet.print_summary(3, step_up=True, show_hierarchical=True)

    # the classes are unbalanced, so we need to artificially increase the
    # weight of the positive classes
    class_weights = get_class_weights()

    # initialize loss function and optimizer
    loss_func = BCEWithLogitsLoss(pos_weight=class_weights.to(DEVICE), reduction='sum')
    opt = RMSprop(unet.parameters(), lr=INIT_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.ReduceLROnPlateau(opt, 'max', patience=5)
    es = EarlyStopping(patience=15, min_delta=0, restore_best_weights=True)
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
    torch.save(unet.state_dict(), model_path)

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
    class_weights = 1. / class_weights
    # class_weights = class_weights / class_weights.sum()

    print("Class weights: ", class_weights)

    class_weights = class_weights.repeat(IMAGE_SIZE, IMAGE_SIZE, 1)
    class_weights = class_weights.permute(2, 0, 1)
    return class_weights
