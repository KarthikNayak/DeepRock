import keras
from keras.layers import Activation
from keras.layers.core import Dense, Dropout
from keras.layers.cudnn_recurrent import CuDNNLSTM
from keras.models import Sequential

from create_data import *


def create_network(input_shape, op_len, lr):
    model = Sequential()
    model.add(CuDNNLSTM(256, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(CuDNNLSTM(256))
    model.add(Dense(128))
    model.add(Dropout(0.3))
    model.add(Dense(op_len, activation="softmax"))

    opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
    return model


def get_xy():
    x, y, =  None, None
    with open('object/data/X', 'rb') as file:
        x = pickle.load(file)
    with open('object/data/Y', 'rb') as file:
        y = pickle.load(file)

    return x, y


def train():
    x, y = get_xy()
    onehot = None
    with open('object/data/onehot', 'rb') as file:
        onehot = pickle.load(file)

    for lr in [0.01, 0.001, 0.0001]:
        print(f"Training with lr: {lr}")
        model = create_network(x.shape[1:], onehot.categories_[0].shape[0], lr)
        history = model.fit(x, y, epochs=200, batch_size=64)
        model_json = model.to_json()
        with open(f"object/models/model{lr}.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(f"object/models/model{lr}.h5")
        with open(f"object/models/history{lr}", 'wb') as f:
            pickle.dump(history, f)

    print("Saved models to disk")


if __name__ == '__main__':
    train()
