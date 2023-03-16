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

    # the classes are unbalanced, so we need to artificially increase the
    # weight of the positive classes
    class_weights = torch.tensor(CLASS_WEIGHTS)
    class_weights = 1. / class_weights
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.repeat(IMAGE_SIZE, IMAGE_SIZE, 1)
    class_weights = class_weights.permute(2, 0, 1)

    # initialize loss function and optimizer
    loss_func = BCEWithLogitsLoss(pos_weight=class_weights.to(DEVICE))
    opt = RMSprop(unet.parameters(), lr=INIT_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.ReduceLROnPlateau(opt, 'max', patience=5)

    metrics = SegmentationMetrics()

    # calculate steps per epoch for training and test set
    train_steps = len(train_ds) // BATCH_SIZE
    test_steps = len(test_ds) // BATCH_SIZE

    # initialize a dictionary to store training history
    H = {"train_loss": [], "test_loss": []}

    # loop over epochs
    print("[INFO] training the network...")
    start_time = time.time()
    for e in tqdm(range(NUM_EPOCHS)):
        # set the model in training mode
        unet.train()

        # initialize the total training and validation loss
        total_train_loss = 0
        total_test_loss = 0

        # loop over the training set
        for (i, (x, y)) in enumerate(train_loader):
            # send the input to the device
            (x, y) = (x.to(DEVICE), y.to(DEVICE))

            # perform a forward pass and calculate the training loss
            logits, _ = unet(x)
            loss = loss_func(logits, y)

            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            opt.zero_grad()
            loss.backward()
            opt.step()

            # add the loss to the total training loss so far
            total_train_loss += loss

        metrics_results = torch.tensor([0.0 for _ in metrics])

        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            unet.eval()

            # loop over the validation set
            for (x, y) in test_loader:
                # send the input to the device
                (x, y) = (x.to(DEVICE), y.to(DEVICE))

                # make the predictions and calculate the validation loss
                logits, masks = unet(x)
                total_test_loss += loss_func(logits, y)

                # calculate the metrics
                metrics_results += torch.tensor([m(masks, y) for m in metrics])

        # calculate the average training and validation loss
        avg_train_loss = total_train_loss / train_steps
        avg_test_loss = total_test_loss / test_steps

        # update the learning rate
        scheduler.step(avg_test_loss)

        # update our training history
        H["train_loss"].append(avg_train_loss.cpu().detach().numpy())
        H["test_loss"].append(avg_test_loss.cpu().detach().numpy())

        # print the model training and validation information
        print(f"[INFO] EPOCH: {e + 1}/{NUM_EPOCHS}")
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(avg_train_loss, avg_test_loss))
        print("Metrics:")
        metrics_results = metrics_results / test_steps
        for i, res in enumerate(metrics):
            print(f"  {res.__name__}: {(metrics_results[i]):.4f}")

        print("[INFO] saving the model...")
        model_path = os.path.join(BASE_OUTPUT, "unet_intermediate.pth")
        torch.save(unet.state_dict(), model_path)

    # display the total time needed to perform the training
    end_time = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        end_time - start_time))

    # save the model
    print("[INFO] saving the model...")
    model_path = os.path.join(BASE_OUTPUT, "unet.pth")
    torch.save(unet.state_dict(), model_path)

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["test_loss"], label="test_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend(loc="lower left")

    loss_plot_path = os.path.join(BASE_OUTPUT, "loss.png")
    plt.savefig(loss_plot_path)

    print("[INFO] Training completed!\n")
