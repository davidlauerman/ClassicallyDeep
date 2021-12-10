# Machine Composed Music using Deep Neural Networks

## Inspiration

When our team came together, we all knew we wanted to do something about music. The idea of being able to generate brand-new melodies with a deep learning model was an absolute pick for us.


## Structure

We have our neural network itself in the class model.py. Our preprocessing/network training is in
classically_deep.py. This means that all of the model-related parameters are found in the model class,
and the data preprocessing and postprocessing parameters are found in the classically_deep file. We
process our data from MusicNet piece by piece in organize_song(), and then concatenate all tokenized
songs into one numpy array, which can then be saved to a file to facilitate future use. Then, the model
is trained by this input data, and then the trained model is used to generate a song using generate_song().
This can then be used to create a MIDI file using generate_MIDI().

## How to Use

To preprocess song data and save it to a file, set save_mode = True in main() of classically_deep.py,
line 212. If you want to load this data and run the model, set save_mode = False. Do note that in order
to run this project, you will need the .npz version of MusicNet, which occupies roughly 11.72 GB of storage.
This project has a number of parameters relating to number of training songs, number of samples to take from
each song, how frequently to take samples, etc. These are global variables and are found at the top of
classically_deep.py. This project expects the MusicNet.npz file to be located in a folder called "data/musicnet.npz"
from the directory that classically_deep.py is located in, which is also where the output song will be written.
