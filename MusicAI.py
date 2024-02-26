import random
import numpy as np
import keras
from pathlib import Path
from music21 import converter, instrument, note, chord
#from keras.utils.module_utils
import tensorflow as tf
from keras.utils import to_categorical
#from keras.utils import np_utils

songs = []
folder = Path("./archive")
for file in folder.rglob('*.mid'):
  songs.append(file)
#for song in songs:
    #print(song)
result =  random.sample([x for x in songs], 93)
#for rSong in result:
    #print("Song in result: ")
    #print(rSong)
notes = []
for i,file in enumerate(result):
    print(f'{i+1}: {file}')
    try:
      midi = converter.parse(file)
      notes_to_parse = None
      parts = instrument.partitionByInstrument(midi)
      if parts: # file has instrument parts
          notes_to_parse = parts.parts[0].recurse()
      else: # file has notes in a flat structure
          notes_to_parse = midi.flat.notes
      for element in notes_to_parse:
          if isinstance(element, note.Note):
              notes.append(str(element.pitch))
          elif isinstance(element, chord.Chord):
              notes.append('.'.join(str(n) for n in element.normalOrder))
    except:
      print(f'FAILED: {i+1}: {file}')

import pickle
with open('notes', 'wb') as filepath:
  pickle.dump(notes, filepath)

  
def prepare_sequences(notes, n_vocab):
    #Prepare the sequences used by the Neural Network
    sequence_length = 32

    # Get all unique pitchnames
    pitchnames = sorted(set(item for item in notes))
    numPitches = len(pitchnames)

     # Create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        # sequence_in is a sequence_length list containing sequence_length notes
        sequence_in = notes[i:i + sequence_length]
        # sequence_out is the sequence_length + 1 note that comes after all the notes in
        # sequence_in. This is so the model can read sequence_length notes before predicting
        # the next one.
        sequence_out = notes[i + sequence_length]
        # network_input is the same as sequence_in but it containes the indexes from the notes
        # because the model is only fed the indexes.
        network_input.append([note_to_int[char] for char in sequence_in])
        # network_output containes the index of the sequence_out
        network_output.append(note_to_int[sequence_out])

    # n_patters is the length of the times it was iterated 
    # for example if i = 3, then n_patterns = 3
    # because network_input is a list of lists
    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    # Reshapes it into a n_patterns by sequence_length matrix
    print(len(network_input))
    
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    # OneHot encodes the network_output
    network_output = to_categorical(network_output)

    return (network_input, network_output)


n_vocab = len(set(notes))
network_input, network_output = prepare_sequences(notes,n_vocab)
n_patterns = len(network_input)
pitchnames = sorted(set(item for item in notes))
numPitches = len(pitchnames)

