import math
import numpy as np
from intervaltree import Interval,IntervalTree
import os
import torch
from model import GAN

# def train(model, train_loader):
#     """
#     Train GAN with one epoch.
#
#     Inputs:
#     - model: a GAN instance.
#     - train_loader: A torch dataset of MusicNet.
#
#     Returns:
#     - total_loss: Sum of loss values of all batches.
#     """
#
#     total_loss = 0
#     optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
#     i = 0
#     print("")
#     for data, labels in train_loader:
#         print(f"training batch \033[0;33m{i +1}\033[0m"), end = "\r")
#         with tf.GradientTape() as tape:
#             if is_cvae:
#                 one_hot_vec = one_hot(labels, 10)
#                 x_hat, mu, logvar = model.call(data, one_hot_vec)
#
#             else:
#                 x_hat, mu, logvar = model.call(data)
#
#             loss = loss_function(x_hat, data, mu, logvar)
#             total_loss += loss
#
#         gradients = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#         i += 1
#
#     print(f"training batch \033[0;32m{i + 1}\033[0m/\033[0;32m{i + 1}\ntraining complete!\033[0m"), end = "\n")
#     return total_loss


def main():

    # suppress messages from torch
    os.environ["PYTORCH_JIT_LOG_LEVEL"] = "2"

    # Load MusicNet dataset from .npz file
    # train_data is a dictionary of arrays, indexed 0-329
    train_data = np.load('data/musicnet.npz', allow_pickle = True, encoding = 'latin1')


    ids = list(train_data.keys())

    samples, labels = train_data['2494']

    print(type(samples))
    print(type(labels))

    # Get an instance of GAN
    # model = GAN();
    #
    # # Train GAN
    # for epoch_id in range(args.num_epochs):
    #     total_loss = train_vae(model, train_dataset, args, is_cvae=args.is_cvae)
    #     print(f"Train Epoch: {epoch_id} \tLoss: {total_loss/len(train_dataset):.6f}")

if __name__ == "__main__":
    main()
