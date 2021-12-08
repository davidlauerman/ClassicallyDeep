import math
import numpy as np
from intervaltree import Interval,IntervalTree
import os
import torch
import random
from model import DistributionLearner,QuantityLearner
#import play

DEFAULT_HZ = 44100												# audio play rate
STEP_SIZE = 2205												# how often to sample (currently every 0.05 sec)
NUM_SEC = 60													# how many seconds to survey from each song
NUM_NOTES = 128													# number of possible MIDI notes
SAMPLES_PER_SONG = int((DEFAULT_HZ / STEP_SIZE) * NUM_SEC)		# number of samples to take per song
NUM_SONGS = 50													# number of songs to train on
NUM_NOTES_OUTPUT = 3											# max number of notes to have in output

def train(dist_model, quant_model, inputs, labels):
	"""
	Train RNN with one epoch.

	Inputs:
	- model: a RNN instance.
	- input: all of the song data in one np array

	Returns:
	None
	"""

	dist_optimizer = torch.optim.Adam(dist_model.parameters(), lr = dist_model.learning_rate)
	quant_optimizer = torch.optim.Adam(quant_model.parameters(), lr = quant_model.learning_rate)

	loss_function = torch.nn.MSELoss()

	dist_hidden_state = torch.randn(dist_model.num_layers, dist_model.batch_size, dist_model.hidden_size)
	quant_hidden_state = torch.randn(quant_model.num_layers, quant_model.batch_size, quant_model.hidden_size)

	w = dist_model.window_size
	b = dist_model.batch_size
	p = w * b


	max = int(inputs.shape[0]/ (dist_model.window_size * dist_model.batch_size))

	for i in range(max):
		print(f"training batch \033[0;33m{i + 1}\033[0m/\033[0;32m{max}\033[0;0m", end = "\r")

		# batch data
		inputs_batch = torch.as_tensor(inputs[i * p: (i + 1) * p, :])
		labels_batch = torch.as_tensor(labels[i * p: (i + 1) * p, :])

		dist_inputs_batch = torch.reshape(inputs_batch, (b, w, dist_model.input_size))
		dist_labels_batch = torch.reshape(labels_batch, (b, w, dist_model.input_size))

		quant_inputs_batch = torch.as_tensor(np.count_nonzero(inputs_batch, axis = 1))
		quant_labels_batch = torch.as_tensor(np.count_nonzero(labels_batch, axis = 1))

		quant_inputs_batch = torch.reshape(quant_inputs_batch, (b, w, quant_model.hidden_size))
		quant_labels_batch = torch.reshape(quant_labels_batch, (b, w, quant_model.hidden_size))


		############################## DIST_MODEL ##############################

		# allow us to update the hidden state without pytorch complaining
		dist_hidden_state = dist_hidden_state.detach()

		# compute probabilities
		dist_output, dist_hidden_state = dist_model.call(dist_inputs_batch, dist_hidden_state)

		# compute loss
		dist_loss = loss_function(dist_output, dist_labels_batch)

		# zero optimizers
		dist_model.zero_grad()

		# apply loss
		dist_loss.backward()
		dist_optimizer.step()

		############################## QUANT_MODEL #############################

		# allow us to update the hidden state without pytorch complaining
		quant_hidden_state = quant_hidden_state.detach()

		# compute probabilities
		quant_output, quant_hidden_state = quant_model.call(quant_inputs_batch, quant_hidden_state)

		# compute loss
		quant_loss = loss_function(quant_output, quant_labels_batch)

		# zero optimizers
		quant_model.zero_grad()

		# apply loss
		quant_loss.backward()
		quant_optimizer.step()

		i += 1

	print(f"training batch \033[0;32m{max}\033[0m/\033[0;32m{max}\ntraining complete!\033[0m", end = "\n")


def organize_song(id, length, song_info):

	# data stores the number of samples + 1,
	#   where the first will be stripped to form labels
	#   and the last will be stripped to form inputs
	data = np.zeros((SAMPLES_PER_SONG + 1, NUM_NOTES))

	# randomize the starting position of the song
	start_point = random.randint(0, length - (DEFAULT_HZ * NUM_SEC + STEP_SIZE))

	# looping through every single time step, getting a data point every STEP_SIZE
	for timestep in range(start_point, start_point + (DEFAULT_HZ * NUM_SEC) + STEP_SIZE, STEP_SIZE):

		# sort the interval tree
		sorted_song_info = sorted(song_info[timestep])

		# extract all playing notes at a given timestep
		for note_package in range(len(song_info[timestep])):
			data[int((timestep - start_point)/STEP_SIZE)][sorted_song_info[note_package][2][1]] = 1

	# returns inputs, labels
	return data[:-1], data[1:]


"""
start_notes: array of starter notes [num_notes x model.hidden_size]
length_song: the length of the generated sequence after the starting notes

"""
def generate_song(dist_model, quant_model, start_notes, length_song):

	output_song = np.zeros((length_song, dist_model.hidden_size))
	dist_hidden_state = None
	quant_hidden_state = None

	# generate hidden state with starter sequence
	for i in range(start_notes.shape[0]):
		output_song[i] = start_notes[i]
		current_input = torch.reshape(torch.as_tensor(start_notes[i]), (1, 1, dist_model.hidden_size))
		next_notes, dist_hidden_state = dist_model.call(current_input, dist_hidden_state)

		next_number, quant_hidden_state = quant_model.call(current_input, quant_hidden_state)

	# generate song
	for i in range(start_notes.shape[0], length_song):
		next_notes = torch.reshape(next_notes, (1, 1, dist_model.hidden_size))
		next_notes, dist_hidden_state = dist_model.call(next_notes, hidden_state)

		next_number, quant_hidden_state = quant_model.call(next_number, quant_hidden_state)

		next_notes = next_notes.detach()

		next_notes = 1 * (np.argsort(np.argsort(next_notes)) >= next_notes.shape[1] - next_number)

		output_song[i] = np.array(torch.reshape(next_notes, (1, dist_model.hidden_size)))

	return output_song



def main():

	# suppress messages from torch
	os.environ["PYTORCH_JIT_LOG_LEVEL"] = "2"

	torch.set_default_tensor_type(torch.DoubleTensor)
	torch.autograd.set_detect_anomaly(True)

	# create random seed based on system time
	random.seed()

	# Load MusicNet dataset from .npz file
	# train_data is a dictionary of arrays, indexed by ids
	raw_data = np.load('data/musicnet.npz', allow_pickle = True, encoding = 'latin1')

	# get ids to iterate through ids
	ids = list(raw_data.keys())

	save_mode = False
	assert 1 <= NUM_SONGS <= 330

	if save_mode:
		i = 0
		for id in ids:
			if i == NUM_SONGS:
				break

			audio, labels = raw_data[id]

			if i == 0:
				print(f"chronicling song \033[93m{i + 1}\033[0m/\033[92m{NUM_SONGS}\033[0m", end = "\r")
				INPUTS, LABELS = organize_song(id, len(audio), labels)
			else:
				print(f"chronicling song \033[93m{i + 1}\033[0m/\033[92m{NUM_SONGS}\033[0m", end = "\r")
				song_inp, song_lab = organize_song(id, len(audio), labels)
				INPUTS = np.vstack((INPUTS, song_inp))
				LABELS = np.vstack((LABELS, song_lab))

			i += 1

		print(f"chronicling song \033[92m{NUM_SONGS}\033[0m/\033[92m{NUM_SONGS}\033[0m", end = "\n")
		print("\033[93msaving song arrays...\033[0m")
		np.savez_compressed(f'data/songs.npz', inputs = INPUTS, labels = LABELS)
		print("\033[92msong arrays saved successfully!\033[0m")

		return

	# otherwise, load data and create model
	print("\033[93mloading song array...\033[0m")
	train_data = np.load('data/songs.npz')
	print("\033[92msong array loaded successfully!\033[0m")

	inputs = train_data['inputs']
	labels = train_data['labels']

	# Get an instance of the two models
	dist_model = DistributionLearner();
	quant_model = QuantityLearner();

	# Train RNN
	train(dist_model, quant_model, inputs, labels)

	rand = random.randint(0, inputs.shape[0] - 33)

	start_notes = inputs[rand : rand + 33]
	length_song = 300

	song = generate_song(dist_model, quant_model, start_notes, length_song)

	currently_masoning = 1 * (np.argsort(np.argsort(song_probabilities)) >= song_probabilities.shape[1] - 3)
	#print(currently_masoning[40])

if __name__ == "__main__":
	main()
