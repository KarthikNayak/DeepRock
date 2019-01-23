import glob
import pickle

import numpy as np
from music21 import chord, converter, instrument, note, stream
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Global variable
sequence_length = 100
data_percent = 100
unique_factor = 13


# Parse the MIDIs to get only tracks with Guitars in them
def populate_guitar_track():
    guitar_parts = []
    for file in glob.glob("midi/**/*.mid", recursive=True):
        try:
            score = converter.parse(file)
            guitar = instrument.ElectricGuitar
            for part in instrument.partitionByInstrument(score):
                if isinstance(part.getInstrument(), guitar):
                    print(f"Has Guitar: {file}")
                    guitar_parts.append(file)
        except:
            continue

    with open('object/data/guitar_midi_files', 'wb') as file:
        pickle.dump(guitar_parts, file)


# Generator to go through Guitar MIDIs and yield the tracks
def get_tracks():
    with open('object/data/guitar_midi_files', 'rb') as f:
        guitar_parts = pickle.load(f)
        for file in guitar_parts:
            print(f"In file: {file}")
            song = converter.parse(file)
            for part in instrument.partitionByInstrument(song):
                if isinstance(part.getInstrument(), instrument.ElectricGuitar):
                    yield part


# Get all notes from a given track
def get_notes(seq_len=1, reset=False):
    data = []
    print(f"Parsing notes with reset set to {reset}")
    if not reset:
        with open('object/data/notes', 'rb') as f:
            print(f"Returning notes from pickle")
            return pickle.load(f)
    for track in get_tracks():
        tmp = []
        notes = track.recurse()
        for n in notes:
            if isinstance(n, note.Note):
                tmp.append(str(n.pitch))
            elif isinstance(n, chord.Chord):
                tmp.append(' '.join(str(x.pitch) for x in n))
        tmp = tmp[:int(len(tmp)/seq_len)*seq_len]
        data.extend(tmp)
    print(f"Done parsing notes")
    with open('object/data/notes', 'wb') as f:
        pickle.dump(data, f)
    return data


# Check if the number of unique notes confirms to our minimum requirement
def check_data(data):
    if len(np.unique(data)) > unique_factor:
        return True
    return False


# Parse notes and creating Training data
def create_training_data(reset=False):
    X = []
    Y = []
    data = get_notes(sequence_length, reset)
    idx = int(len(data) * data_percent/100)

    enc = OrdinalEncoder()
    enc.fit(np.array(data).reshape(-1, 1))

    print(f"Creating data from notes of size: {len(data)}")

    for i in range(0, idx - sequence_length):
        if check_data(data[i:i+sequence_length]):
            X.append(enc.transform(np.reshape(
                data[i:i+sequence_length], (-1, 1))))
            Y.append(enc.transform(np.reshape(
                data[i+sequence_length], (-1, 1))))

    X = np.array(X)
    Y = np.array(Y)

    mean = X.mean()
    std = X.std()
    X = (X - mean) / std

    onehot = OneHotEncoder(sparse=False)
    Y = onehot.fit_transform(Y.reshape(-1, 1))

    with open('object/data/parameters', 'wb') as file:
        pickle.dump({'mean': mean, 'std': std}, file)
    with open('object/data/enc', 'wb') as file:
        pickle.dump(enc, file)
    with open('object/data/onehot', 'wb') as file:
        pickle.dump(onehot, file)
    with open('object/data/X', 'wb') as file:
        pickle.dump(X, file)
    with open('object/data/Y', 'wb') as file:
        pickle.dump(Y, file)

    print("Created X, Y")


if __name__ == '__main__':
    create_training_data(reset=False)
