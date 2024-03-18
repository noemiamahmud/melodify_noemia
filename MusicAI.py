import random
import numpy as np
import keras
from pathlib import Path
from music21 import converter, instrument, note, chord
#from keras.utils.module_utils
import tensorflow as tf
from keras.utils import to_categorical
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.callbacks import ModelCheckpoint
#from keras.utils import np_utils

def load_notes():
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
    return notes


def preocess_notes(notes):

    dataSize = 150000
    allNotes = allNotes[:dataSize]

    # Look at 10 previous notes to make a prediction
    #   We can tune this parameter if needed, based on the length of 
    #   chord progressions
    seqLength = 10
    print('Using sequence length of {}'.format(seqLength))

    pitchSet = sorted(set(allNotes))
    numPitches = len(pitchSet)
    # here pitches are either notes or chords
    #   they are sorted lexicographically, so a chord 'C4.E4' will come after a
    #   note 'C4'
    print('Identified {} pitches'.format(numPitches))

    # Map each note/chord to a number normalized to (0,1)
    pitchMapping = dict((note, number) for (number, note) in enumerate(pitchSet))

    networkInput = []
    networkOutput = []

    print('Starting sequencing of {} notes'.format(len(allNotes)))
    for i in range(0, len(allNotes)- seqLength):
        sequenceIn = allNotes[i:i+seqLength]
        predictionOut = allNotes[i+seqLength]

        networkInput.append([pitchMapping[note] for note in sequenceIn])
        networkOutput.append(pitchMapping[predictionOut])

        if (i+1) % 400000 == 0:
            print('Finished making {} sequences'.format(i+1))

    networkInput = np.array(networkInput)
    networkOutput = np.array(networkOutput)

    numSeqs = len(networkInput)
    # reshape input to match the LSTM layer format
    networkInputShaped = np.reshape(networkInput, (numSeqs, seqLength, 1))
    networkInputShaped = networkInputShaped / numPitches

    networkOutputShaped = to_categorical(networkOutput)

    return networkInput, networkOutput, networkInputShaped, networkOutputShaped, numPitches

def create_model(networkInputShaped,networkOutputShaped,numPitches,num_epochs=30):
    model = Sequential()
    model.add(Dropout(0.2))
    model.add(LSTM(
        512,
        input_shape=(networkInputShaped.shape[1], networkInputShaped.shape[2]),
        return_sequences=True
    ))
    model.add(Dense(256))
    model.add(Dense(256))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dense(256))
    model.add(LSTM(512))
    model.add(Dense(numPitches))
    model.add(Dense(numPitches))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    num_epochs = 30

    filepath = "/weights/weights-improvement-{epoch:02d}-{loss:.4f}-bigger_1.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor='loss', 
        verbose=1,        
        save_best_only=True,        
        mode='min'
    )    
    callbacks_list = [checkpoint]


    history = model.fit(networkInputShaped, networkOutputShaped, epochs=num_epochs, batch_size=64, callbacks=callbacks_list)

    return model, history

def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=5, batch_size=128, callbacks=callbacks_list)

    return model

def generate_notes(model, network_input, pitchnames, n_vocab):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    # Selects a random row from the network_input
    start = np.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    # Random row from network_input
    pattern = network_input[start]
    prediction_output = []

    # generate 500 notes
    for note_index in range(500):
        print(note_index)
        # Reshapes pattern into a vector
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        # Standarizes pattern
        prediction_input = prediction_input / float(n_vocab)

        # Predicts the next note
        prediction = model.predict(prediction_input, verbose=0)

        # Outputs a OneHot encoded vector, so this picks the columns
        # with the highest probability
        index = np.argmax(prediction)
        # Maps the note to its respective index
        result = int_to_note[index]
        # Appends the note to the prediction_output
        prediction_output.append(result)

        # Adds the predicted note to the pattern
        pattern = np.append(pattern,index)
        # Slices the array so that it contains the predicted note
        # eliminating the first from the array, so the model can
        # have a sequence
        pattern = pattern[1:len(pattern)]

    return prediction_output

from music21 import instrument, note, stream

def create_midi(prediction_output):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='test_output.mid')

if __name__ == '__main__':
    notes = load_notes()

    n_vocab = len(set(notes))
    pitchnames = sorted(set(item for item in notes))

    networkInput, networkOutput, networkInputShaped, networkOutputShaped, numPitches = preocess_notes(notes)
    model, history = create_model(networkInputShaped,networkOutputShaped,numPitches,num_epochs=30)
    prediction_output = generate_notes(model, networkInputShaped, pitchnames, n_vocab)

    create_midi(prediction_output)

