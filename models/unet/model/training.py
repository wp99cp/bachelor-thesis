import json
import os
import time

import torch
from matplotlib import pyplot as plt
from pytorch_model_summary import summary
from torch.nn import BCEWithLogitsLoss
from torch.optim import RMSprop, lr_scheduler
from tqdm import tqdm

from configs.config import DEVICE, INIT_LR, BATCH_SIZE, NUM_EPOCHS, BASE_OUTPUT, IMAGE_SIZE, CLASS_WEIGHTS, MOMENTUM, \
    WEIGHT_DECAY, NUM_CHANNELS
from model.Model import UNet
from model.metrices import SegmentationMetrics


def training(train_loader, test_loader, train_ds, test_ds):
    # initialize our UNet model
    unet = UNet().to(DEVICE)

    # print the model summary
    print_model_summary(unet)

    # the classes are unbalanced, so we need to artificially increase the
    # weight of the positive classes
    class_weights = get_class_weights()

    print(f"[INFO] class weights: {class_weights}")

    # initialize loss function and optimizer
    loss_func = BCEWithLogitsLoss(pos_weight=class_weights.to(DEVICE), reduction='sum')
    opt = RMSprop(unet.parameters(), lr=INIT_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.ReduceLROnPlateau(opt, 'max', patience=5)

    metrics = SegmentationMetrics()

    # calculate steps per epoch for training and test set
    train_steps = len(train_ds) // BATCH_SIZE
    test_steps = len(test_ds) // BATCH_SIZE

    # initialize a dictionary to store training history
    history = create_empty_history(metrics)

    # loop over epochs
    print("[INFO] training the network...")
    start_time = time.time()
    for e in tqdm(range(NUM_EPOCHS)):
        print(f"[INFO] EPOCH: {e + 1}/{NUM_EPOCHS}")

        train_loss = train_model(model=unet, loss_func=loss_func, optimizer=opt, loader=train_loader,
                                 num_batches=train_steps, history=history)
        test_loss = eval_model(model=unet, loss_func=loss_func, metrics=metrics, loader=test_loader,
                               num_batches=test_steps, history=history)

        print("Train loss: {:.6f}, Test loss: {:.4f}".format(train_loss, test_loss))

        # update the learning rate
        scheduler.step(test_loss)

        print("[INFO] saving the model...")
        model_path = os.path.join(BASE_OUTPUT, "unet_intermediate.pth")
        torch.save(unet.state_dict(), model_path)

    # display the total time needed to perform the training
    end_time = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s = {:.2f}m = {:.2f}h".format(end_time - start_time, (
            end_time - start_time) / 60, (end_time - start_time) / 3600))

    # save history as JSON
    print("[INFO] saving the training history...")

    for key, value in history.items():
        history[key] = [v.tolist() for v in value]

    history_file = open(os.path.join(BASE_OUTPUT, "history.json"), "w")
    history_file.write(json.dumps(history))
    history_file.close()

    # save the model
    print("[INFO] saving the model...")
    model_path = os.path.join(BASE_OUTPUT, "unet.pth")
    torch.save(unet.state_dict(), model_path)

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

    print("[INFO] Training completed!\n")


def create_empty_history(metrics):
    history = {"train_loss": [], "test_loss": []}
    # add metrics to history
    for metric in metrics:
        history[metric.__name__] = []
    return history


def get_class_weights():
    class_weights = torch.tensor(CLASS_WEIGHTS)
    class_weights = 1. / class_weights
    # class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.repeat(IMAGE_SIZE, IMAGE_SIZE, 1)
    class_weights = class_weights.permute(2, 0, 1)
    return class_weights


def print_model_summary(unet):
    print(summary(unet, torch.zeros((1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)).to(DEVICE),
                  show_input=True,
                  max_depth=1))
    print(summary(unet, torch.zeros((1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)).to(DEVICE),
                  show_input=True,
                  max_depth=2))
    print(summary(unet, torch.zeros((1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)).to(DEVICE),
                  show_input=True,
                  max_depth=3,
                  show_hierarchical=True))


def eval_model(model, loss_func, metrics, loader, num_batches, history):
    metrics_results = torch.tensor([0.0 for _ in metrics])

    total_test_loss = 0

    # switch off autograd
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()

        # loop over the validation set
        for (x, y) in loader:
            # send the input to the device
            (x, y) = (x.to(DEVICE), y.to(DEVICE))

            # make the predictions and calculate the validation loss
            logits, masks = model(x)
            total_test_loss += loss_func(logits, y)

            # calculate the metrics
            metrics_results += torch.tensor([m(masks, y) for m in metrics])

    avg_test_loss = total_test_loss / num_batches

    # update our training history
    history["test_loss"].append(avg_test_loss.cpu().detach().numpy())

    # print the model training and validation information

    print("Metrics:")
    metrics_results = metrics_results / num_batches
    for i, res in enumerate(metrics):
        print(f"  {res.__name__}: {(metrics_results[i]):.4f}")
        history[res.__name__].append(metrics_results[i].cpu().detach().numpy())

    return avg_test_loss


def train_model(model, loss_func, optimizer, loader, num_batches, history):
    # set the model in training mode
    model.train()

    # initialize the total training and validation loss
    total_train_loss = 0

    # loop over the training set
    for (i, (x, y)) in enumerate(loader):
        # send the input to the device
        (x, y) = (x.to(DEVICE), y.to(DEVICE))

        # perform a forward pass and calculate the training loss
        logits, _ = model(x)
        loss = loss_func(logits, y)

        # first, zero out any previously accumulated gradients, then
        # perform backpropagation, and then update model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # add the loss to the total training loss so far
        total_train_loss += loss

    # calculate the average training and validation loss
    avg_train_loss = total_train_loss / num_batches
    history["train_loss"].append(avg_train_loss.cpu().detach().numpy())

    return avg_train_loss
