import math
import numpy as np
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
#
# def load_musicnet(batch_size, buffer_size=1024):
#     """
#     Load and preprocess MNIST dataset from tf.keras.datasets.mnist.
#
#     Inputs:
#     - batch_size: An integer value of batch size.
#     - buffer_size: Buffer size for random sampling in tf.data.Dataset.shuffle().
#
#     Returns:
#     - train_dataset: A tf.data.Dataset instance of MNIST dataset. Batching and shuffling are already supported.
#     """
#     mnist = tf.keras.datasets.mnist
#     (x_train, y_train), _ = mnist.load_data()
#     x_train = x_train / 255.0
#     x_train = np.expand_dims(x_train, axis=1)  # [batch_sz, channel_sz, height, width]
#     train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#     train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)
#     return train_dataset

def main():

    x = torch.rand(5, 3)
    print(x)

    # # suppress messages from tf
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    #
    # # Load MusicNet dataset
    # train_dataset = load_musicnet(args.batch_size)
    #
    # # Get an instance of GAN
    # model = GAN();
    #
    # # Train GAN
    # for epoch_id in range(args.num_epochs):
    #     total_loss = train_vae(model, train_dataset, args, is_cvae=args.is_cvae)
    #     print(f"Train Epoch: {epoch_id} \tLoss: {total_loss/len(train_dataset):.6f}")

if __name__ == "__main__":
    main()
