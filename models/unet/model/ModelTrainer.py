import os
from signal import signal, SIGUSR1

import torch
import tqdm
from configs.config import NUM_EPOCHS, BATCH_SIZE, BASE_OUTPUT, DEVICE
from model import Model
from model.EarlyStopping import EarlyStopping
from pytictac import ClassTimer, accumulate_time
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.unet.configs.config import STEPS_PER_EPOCH, STEPS_PER_EPOCH_TEST


class ModelTrainer:
    """
    This class is responsible for training the model.
    """

    def __init__(self,
                 model: Model,
                 loss_func: _Loss,
                 optimizer: Optimizer,
                 scheduler: ReduceLROnPlateau,
                 early_stopping: EarlyStopping,
                 metrics: list):

        self.epoch = 0
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.metrics = metrics

        self.history = None
        self.initialize_history()

        self.cct = ClassTimer(objects=[self], names=["ModelTrainer"], enabled=True)

        # We register a signal handler for SIGUSR1 to interrupt the training
        # and save the model.
        self.emergency_stop = False
        signal(SIGUSR1, self.__interrupt_handler)

    @accumulate_time
    def train(self, train_ds, test_ds, train_loader, test_loader, num_epochs=NUM_EPOCHS):

        print('start training: trigger an emergency stop with: "kill -SIGUSR1 {}"'.format(os.getpid()))

        self.epoch = 0

        # calculate steps per epoch for training and test set
        train_steps = min(len(train_ds) // BATCH_SIZE, STEPS_PER_EPOCH)
        test_steps = min(len(test_ds) // BATCH_SIZE, STEPS_PER_EPOCH_TEST)

        done = False

        while self.epoch < num_epochs and not done and not self.emergency_stop:
            self.epoch += 1

            print(f"\nStart training epoch {self.epoch}...")

            # TODO: forward train ds to switch dates after each step
            train_loss = self.__train_epoch(loader=train_loader, num_batches=train_steps)
            test_loss, metrics_results = self.__test_epoch(loader=test_loader, num_batches=test_steps)

            print(f"Epoch: {self.epoch}, train_loss: {train_loss:>4f}, "
                  f"test_loss: {test_loss:>4f}, EStop: {self.early_stopping.status}")

            self.print_metrics(metrics_results)

            # check if early stopping criteria are met
            done = self.early_stopping(self.model, test_loss)

            # update the learning rate
            self.scheduler.step(test_loss)

            print("[INFO] saving the model...")
            model_path = os.path.join(BASE_OUTPUT, "unet_intermediate.pth")
            torch.save(self.model.state_dict(), model_path)

    def validate(self):
        pass

    def print_metrics(self, metrics_results):
        print("Metrics for validation of epoch: ", self.epoch)
        for i, res in enumerate(self.metrics):
            print(f" - {res.__name__}: {(metrics_results[i]):.4f}")
            self.history[res.__name__].append(metrics_results[i].cpu().detach().numpy())

    @accumulate_time
    def __test_epoch(self, loader, num_batches):
        metrics_results = torch.tensor([0.0 for _ in self.metrics])

        total_test_loss = 0

        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            self.model.eval()

            # loop over the validation set
            for i, (x, y) in enumerate(loader):
                # send the input to the device
                (x, y) = (x.to(DEVICE), y.to(DEVICE))

                # make the predictions and calculate the validation loss
                logits, masks = self.model(x)
                total_test_loss += self.loss_func(logits, y)

                # calculate the metrics
                metrics_results += torch.tensor([m(masks, y) for m in self.metrics])

                if i >= num_batches:
                    break

        avg_test_loss = total_test_loss / num_batches

        # update our training history
        self.history["test_loss"].append(avg_test_loss.cpu().detach().numpy())

        metrics_results = metrics_results / num_batches
        return avg_test_loss, metrics_results

    @accumulate_time
    def __train_epoch(self, loader, num_batches):
        # set the model in training mode
        self.model.train()

        # initialize the total training and validation loss
        total_train_loss = 0
        pbar = tqdm.tqdm(loader)

        # loop over the training set
        for i, (x, y) in enumerate(pbar):
            # send the input to the device
            (x, y) = (x.to(DEVICE), y.to(DEVICE))

            # perform a forward pass and calculate the training loss
            logits, _ = self.model(x)
            loss = self.loss_func(logits, y)

            pbar.set_description(f"Epoch: {self.epoch}, train_loss {loss:}")

            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # add the loss to the total training loss so far
            total_train_loss += loss

            if i >= num_batches:
                break

            if self.emergency_stop:
                print("Emergency stop triggered, skip rest of epoch...")
                break

        # calculate the average training and validation loss
        avg_train_loss = total_train_loss / num_batches
        pbar.set_description(f"Epoch: {self.epoch}, train_loss {avg_train_loss:}")

        self.history["train_loss"].append(avg_train_loss.cpu().detach().numpy())

        return avg_train_loss

    def get_history(self):
        return self.history

    def __interrupt_handler(self, signum, frame):

        print("Signal handler called with signal ", signum)
        self.emergency_stop = True

    def initialize_history(self):
        history = {"train_loss": [], "test_loss": []}
        # add metrics to history
        for metric in self.metrics:
            history[metric.__name__] = []
        self.history = history

    def get_timing_summary(self):
        return f"\nTiming Summary:\n{self.cct.__str__()}\n"
