from train import *


def new_notes(model, starting_notes, mean, std, length):
    notes = starting_notes
    final_notes = notes

    onehot = None
    with open('object/data/onehot', 'rb') as file:
        onehot = pickle.load(file)

    for i in range(length):
        new_note = model.predict(notes)
        argmax = np.argmax(new_note[0])

        val = np.zeros(onehot.categories_[0].shape[0])
        val[argmax] = 1
        val = onehot.inverse_transform(val.reshape(1, -1))
        val = (val - mean)/std

        notes = np.append(notes[0, 1:], val).reshape(
            notes.shape[0], notes.shape[1], notes.shape[2])
        final_notes = np.append(final_notes, val).reshape(
            notes.shape[0], -1, notes.shape[2])
    return final_notes


def store_to_midi(mean, std, notes, filename):
    offset = 0
    output_notes = []

    enc = None
    with open('object/data/enc', 'rb') as file:
        enc = pickle.load(file)

    for row in notes.reshape(-1, 1):
        q = int(round((row[0] * std) + mean))
        p = enc.inverse_transform(np.reshape(q, (-1, 1)))[0][0]
        if ' ' in p:
            notes = p.split(' ')
            chord_notes = []
            for cur_note in notes:
                new_note = note.Note(cur_note)
                new_note.storedInstrument = instrument.ElectricGuitar()
                chord_notes.append(new_note)
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(p)
            new_note.offset = offset
            new_note.storedInstrument = instrument.ElectricGuitar()
            output_notes.append(new_note)
        offset += 0.3
    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp=filename + '.mid')


def generate_music(count):
    json_file = open('object/models/model0.01.json', 'r')
    model = keras.models.model_from_json(json_file.read())
    json_file.close()

    # load weights into new model
    model.load_weights("object/models/model0.01.h5")

    print("Loaded your model")
    print(model.summary())

    X, Y = get_xy()

    parameters = None
    with open('object/data/parameters', 'rb') as f:
        parameters = pickle.load(f)

    for i in range(count):
        rand = np.random.randint(len(X))
        rand = int(rand/sequence_length)*sequence_length
        print('random number: ', rand)
        starting_notes = X[rand].reshape(1, X.shape[1], X.shape[2])

        # Do Prediction
        print(f"Generating music number: {i}")
        pred = new_notes(model, starting_notes,
                         parameters['mean'], parameters['std'], 100)
        # Store with prediction
        store_to_midi(parameters['mean'], parameters['std'],
                      pred, f"music/generation{i}")
    print("Done!:)")


if __name__ == '__main__':
    generate_music(10)
