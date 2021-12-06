import math
import numpy as np
from intervaltree import Interval,IntervalTree
import os
import torch
import random
from model import RNN

DEFAULT_HZ = 44100
STEP_SIZE = 4410
WINDOW_SIZE = 50
STRIDE_SIZE = 50
NUM_SEC = 30
NUM_NOTES = 128
SAMPLES_PER_SONG = int((DEFAULT_HZ / STEP_SIZE) * NUM_SEC)
NUM_SONGS = 100

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

def organize_song(id, length, labels):

    song = np.zeros((SAMPLES_PER_SONG, NUM_NOTES))

    start_point = random.randint(0, length - DEFAULT_HZ * NUM_SEC)

    # looping through every single time step
    for timestep in range(start_point, start_point + DEFAULT_HZ * NUM_SEC, STEP_SIZE):

        sorted_labels = sorted(labels[timestep])

        # extract all playing notes at a given timestep
        for note_package in range(len(labels[timestep])):
            song[int((timestep - start_point)/STEP_SIZE)][sorted_labels[note_package][2][1]] = 1

    return song




def main():

    # suppress messages from torch
    os.environ["PYTORCH_JIT_LOG_LEVEL"] = "2"

    # create random seed based on system time
    random.seed()

    # Load MusicNet dataset from .npz file
    # train_data is a dictionary of arrays, indexed by ids
    train_data = np.load('data/musicnet.npz', allow_pickle = True, encoding = 'latin1')

    # get ids to iterate through ids
    ids = list(train_data.keys())

    save_mode = True
    assert 1 <= NUM_SONGS <= 330

    if save_mode:
        i = 0
        for id in ids:
            if i == NUM_SONGS:
                break

            audio, labels = train_data[id]

            if i == 0:
                print(f"chronicling song \033[93m{i + 1}\033[0m/\033[92m{NUM_SONGS}\033[0m", end = "\r")
                SONGS = organize_song(id, len(audio), labels)
            else:
                print(f"chronicling song \033[93m{i + 1}\033[0m/\033[92m{NUM_SONGS}\033[0m", end = "\r")
                SONGS = np.vstack((SONGS, organize_song(id, len(audio), labels)))

            i = i + 1

        print(f"chronicling song \033[92m{NUM_SONGS}\033[0m/\033[92m{NUM_SONGS}\033[0m", end = "\n")
        print("saving song array...")
        np.savez_compressed(f'data/songs.npz', arr = SONGS)
        print("song array saved successfully!")

        return

    # otherwise, load data and create model
    print("loading song array...")
    data = np.load('data/songs.npz')
    print("song array loaded successfully")

    print(data['arr'].shape)





    # Get an instance of RNN
    # model = RNN();
    #
    # # Train RNN
    # for epoch_id in range(args.num_epochs):
    #     total_loss = train(model, train_dataset, args)
    #     print(f"Train Epoch: {epoch_id} \tLoss: {total_loss/len(train_dataset):.6f}")

if __name__ == "__main__":
    main()
