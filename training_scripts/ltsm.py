import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as tf_layers  # dense, ltsm
from tensorflow.keras.utils import to_categorical


def read_file(filepath: str) -> str:
    with open(filepath, "r") as f:
        text = f.read()
    return text


def generate_text(model, start_string, num_generate):
    input_eval = [char_to_idx[s] for s in start_string]
    input_eval = np.array(input_eval).reshape((1, len(input_eval), 1))

    text_generated = []

    for i in range(num_generate):
        predictions = model.predict(input_eval)
        predicted_id = np.argmax(predictions[-1])

        input_eval = np.append(input_eval[:, 1:], [[predicted_id]], axis=1)
        text_generated.append(idx_to_char[predicted_id])

    return start_string + "".join(text_generated)


text = read_file("training_data/chain_of_thought_cleaned.txt")
chars = sorted(set(text))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

sequence_length = 3
x = []
y = []
for i in range(len(text) - sequence_length):
    x.append([char_to_idx[char] for char in text[i : i + sequence_length]])
    y.append(char_to_idx[text[i + sequence_length]])

x = np.array(x)
y = to_categorical(y, num_classes=len(chars))

# LTSM expects input to be in the shape (num of seqs, seq len, num of feat)
x = x.reshape((x.shape[0], x.shape[1], 1))

# defining the LTSM model
model = Sequential()
model.add(tf_layers.LSTM(50, input_shape=(sequence_length, 1)))
model.add(tf_layers.Dense(len(chars), activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy")

model.fit(x, y, epochs=200, verbose=1)

start_string = "resp"
generated_text = generate_text(model, start_string, 15)
print("Generated text:")
print(generated_text)
