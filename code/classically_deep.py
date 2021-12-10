import math
import numpy as np
from intervaltree import Interval,IntervalTree
import os
import torch
import random
from model import DistributionLearner
import midiutil
import matplotlib.pyplot as plt

DEFAULT_HZ = 44100												# audio play rate
STEP_SIZE = 2205												# how often to sample (currently every 0.05 sec)
																	# should be a factor of DEFAULT_HZ!
NUM_SEC = 60													# how many seconds to survey from each song
NUM_NOTES = 128													# number of possible MIDI notes
SAMPLES_PER_SONG = int((DEFAULT_HZ / STEP_SIZE) * NUM_SEC)		# number of samples to take per song
NUM_SONGS = 300													# number of songs to train on
NUM_NOTES_OUTPUT = 3											# max number of notes to have in one line of output

def train(model, inputs, labels):
	"""
	Train RNN with one epoch.

	Inputs:
	- model: a RNN instance.
	- input: all of the song data in one np array
	- labels: all of the song labels in one np array (inputs shifted a timestep)

	Returns:
	a list of losses
	"""

	# declare optimizer and loss_function
	optimizer = torch.optim.Adam(model.parameters(), lr = model.learning_rate)
	loss_function = torch.nn.BCELoss()

	# setup loss list and initial hidden state
	loss_list = []
	hidden_state = torch.randn(model.num_layers, model.batch_size, model.hidden_size)

	w = model.window_size
	b = model.batch_size
	p = w * b

	max = int(inputs.shape[0]/ p)

	for i in range(max):
		print(f"training batch \033[0;33m{i + 1}\033[0m/\033[0;32m{max}\033[0;0m", end = "\r")

		# batch data
		inputs_batch = torch.as_tensor(inputs[i * p: (i + 1) * p, :])
		labels_batch = torch.as_tensor(labels[i * p: (i + 1) * p, :])

		inputs_batch = torch.reshape(inputs_batch, (b, w, model.input_size))
		labels_batch = torch.reshape(labels_batch, (b, w, model.input_size))

		# allow us to update the hidden state without pytorch complaining
		hidden_state = hidden_state.detach()

		# compute probabilities
		output, hidden_state = model.call(inputs_batch, hidden_state)

		# compute loss
		loss = loss_function(output, labels_batch)

		# zero optimizers
		model.zero_grad()

		# apply loss
		loss.backward()
		optimizer.step()

		loss_list.append(loss)


	print(f"training batch \033[0;32m{max}\033[0m/\033[0;32m{max}\ntraining complete!\033[0m", end = "\n")

	return loss_list


def organize_song(length, song_info):
	"""
		length: the length of the current song
		song_info: all of the note data and metadata, store in an IntervalTree

	"""

	# data stores the number of samples + 1,
	#   where the first will be stripped to form labels
	#   and the last will be stripped to form inputs
	data = np.zeros((SAMPLES_PER_SONG + 1, NUM_NOTES))

	# randomize the starting position of the song
	start_point = length - (DEFAULT_HZ * NUM_SEC + STEP_SIZE)
	if start_point < 0:
		start_point = 0
	start_point = random.randint(0, start_point)

	# looping through every single time step, getting a data point every STEP_SIZE
	for timestep in range(start_point, start_point + (DEFAULT_HZ * NUM_SEC) + STEP_SIZE, STEP_SIZE):

		# sort the interval tree
		sorted_song_info = sorted(song_info[timestep])

		# extract all playing notes at a given timestep
		for note_package in range(len(song_info[timestep])):
			data[int((timestep - start_point)/STEP_SIZE)][sorted_song_info[note_package][2][1]] = 1

	# returns inputs, labels
	return data[:-1], data[1:]


def generate_song(model, start_notes, length_song):
	"""
	start_notes: array of starter notes [num_notes x model.hidden_size]
	length_song: the length of the generated sequence after the starting notes

	"""

	# we'll fill in output_song with the seed followed by the output song
	output_song = np.zeros((length_song, model.hidden_size))
	hidden_state = None

	# generate hidden state with starter sequence
	for i in range(start_notes.shape[0]):
		output_song[i] = start_notes[i]
		current_input = torch.reshape(torch.as_tensor(start_notes[i]), (1, 1, model.hidden_size))
		next_notes, hidden_state = model.call(current_input, hidden_state)

	# generate song
	for i in range(start_notes.shape[0], length_song):
		next_notes = torch.reshape(next_notes, (1, 1, model.hidden_size))
		next_notes, hidden_state = model.call(next_notes, hidden_state)

		next_notes = next_notes.detach()

		# get top NUM_NOTES_OUTPUT many notes from each output line
		output_notes = 1 * (np.argsort(np.argsort(next_notes)) >= next_notes.shape[1] - NUM_NOTES_OUTPUT)

		output_song[i] = np.array(torch.reshape(output_notes, (1, model.hidden_size)))

	return output_song



def generate_MIDI(song):
	"""
		song: a song output array
	"""

	# create MIDI file with 3 tracks
	myMIDI = midiutil.MIDIFile(3)

	# MIDI parameters
	track = 0
	channel = 0
	time = 0
	tempo = 60
	duration = 0.05
	volume = 100

	myMIDI.addTempo(track, time, tempo)

	# iterate through all timesteps in song output
	for timestep in range(0, song.shape[0]):
		current_notes = np.nonzero(song[timestep])

		# iterate through all notes at current timestep, adding to MIDI
		if current_notes[0].size != 0:
			for note in range(len(current_notes)):
				pitch = current_notes[0][note]

				myMIDI.addNote(track, channel, pitch, time, duration, volume)
		time += duration

	# write MIDI to file
	with open("output.mid", "wb") as output_file:
		myMIDI.writeFile(output_file)

def visualize_data(loss_list):
	"""
		loss_list: a list of model losses from training
	"""

	x_values = np.arange(0, len(loss_list), 1)
	y_values = loss_list
	plt.plot(x_values, y_values)
	plt.xlabel('Batch')
	plt.ylabel('Loss')
	plt.title('Loss per Batch Over Time')
	plt.grid(True)
	plt.savefig("loss.png")

def main():

	# suppress messages from torch
	os.environ["PYTORCH_JIT_LOG_LEVEL"] = "2"

	# set default tensor type
	torch.set_default_tensor_type(torch.DoubleTensor)

	# create random seed based on system time
	random.seed()

	# Load MusicNet dataset from .npz file
	# train_data is a dictionary of arrays, indexed by ids
	raw_data = np.load('data/musicnet.npz', allow_pickle = True, encoding = 'latin1')

	# get ids to iterate through ids
	ids = list(raw_data.keys())

	save_mode = True
	assert 1 <= NUM_SONGS <= 330

	# if save_mode is True, then save the data to a file before running the model
	# This MUST be done on the first running of the project
	# Every time the song-saving parameters are altered in this file, save_mode
	# must be re-run
	if save_mode:
		i = 0
		for id in ids:
			if i == NUM_SONGS:
				break

			audio, labels = raw_data[id]

			if i == 0:
				print(f"chronicling song \033[93m{i + 1}\033[0m/\033[92m{NUM_SONGS}\033[0m", end = "\r")
				INPUTS, LABELS = organize_song(len(audio), labels)
			else:
				print(f"chronicling song \033[93m{i + 1}\033[0m/\033[92m{NUM_SONGS}\033[0m", end = "\r")
				song_inp, song_lab = organize_song(len(audio), labels)
				INPUTS = np.vstack((INPUTS, song_inp))
				LABELS = np.vstack((LABELS, song_lab))

			i += 1

		print(f"chronicling song \033[92m{NUM_SONGS}\033[0m/\033[92m{NUM_SONGS}\033[0m", end = "\n")
		print("\033[93msaving song arrays...\033[0m")
		np.savez_compressed(f'data/songs.npz', inputs = INPUTS, labels = LABELS)
		print("\033[92msong arrays saved successfully!\033[0m")


	# otherwise, load data and create model
	print("\033[93mloading song array...\033[0m")
	train_data = np.load('data/songs.npz')
	print("\033[92msong array loaded successfully!\033[0m")

	inputs = train_data['inputs']
	labels = train_data['labels']

	# Get an instance of the two models
	model = DistributionLearner();

	# Train RNN
	loss_list = train(model, inputs, labels)

	# visualize losses
	visualize_data(loss_list)

	# pick random set of notes from input to act as a seed for generation
	rand = random.randint(0, inputs.shape[0] - 33)
	start_notes = inputs[rand : rand + 33]
	length_song = 300

	# generate the output song iteratively using trained RNN
	song = generate_song(model, start_notes, length_song)
	generate_MIDI(song)

if __name__ == "__main__":
	main()
