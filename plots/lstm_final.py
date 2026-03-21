import math
from typing import Iterator, Tuple

import tensorflow as tf
import glob
import os
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import CSVLogger


def build_model(num_layers: int, nodes_per_layer: int) -> Model:
    model = Sequential()

    # embedding layer
    model.add(
        Embedding(
            input_dim=total_words,
            output_dim=100,
            input_length=max_sequence_len - 1,
        )
    )

    for i in range(num_layers):
        if i == num_layers - 1:
            return_seq = False
        else:
            return_seq = True
        model.add(
            LSTM(
                nodes_per_layer,
                return_sequences=return_seq,
            )
        )

    model.add(Dense(total_words, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def generate_chatbot_response(
    input_text: str, next_words: int, model: Model
) -> str:
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([input_text])[0]
        token_list = pad_sequences(
            [token_list], maxlen=max_sequence_len - 1, padding="pre"
        )
        predicted_probs = model.predict(token_list, verbose=1)
        predicted_index = np.argmax(predicted_probs, axis=-1)[0]

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        input_text += " " + output_word
    return input_text


def sequence_generator() -> Iterator[Tuple[int, int]]:
    while True:
        for line in corpus_lines:
            token_list = tokenizer.texts_to_sequences([line])[0]

            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[: i + 1]
                padded_sequence = pad_sequences(
                    [n_gram_sequence], maxlen=max_sequence_len, padding="pre"
                )[0]

                X = padded_sequence[:-1]
                y = to_categorical(
                    padded_sequence[-1], num_classes=total_words
                )

                yield X, y


# main code here
csv_logger = CSVLogger("final_training_log.csv")

corpus_path = "cleaned_data/*.txt"

corpus = ""
max_sequence_len = 0
total_sequences = 0

for file in glob.glob(corpus_path):
    with open(file, "r", encoding="utf-8") as f:
        corpus += f.read().lower() + "\n"  # clean the data while we read it in

# different sizes for testing
# tiny_corpus = corpus[:1000]
# small_corpis = corpus[:10000]
# medium_corpus = corpus[:100000]
# large_corpus = corpus[:500000]


tokenizer = Tokenizer()
corpus_lines = corpus.split("\n")

tokenizer.fit_on_texts(corpus_lines)

split_index = int(len(corpus_lines) * 0.8)
train_lines = corpus_lines[:split_index]
val_lines = corpus_lines[split_index:]

total_words = len(tokenizer.word_index) + 1

for line in corpus_lines:
    tokens = tokenizer.texts_to_sequences([line])[0]
    num_tokens = len(tokens)
    if len(tokens) > max_sequence_len:
        max_sequence_len = num_tokens
    if num_tokens > 1:
        total_sequences += num_tokens - 1

# data is too large to pass in all at once, need to use dataset
dataset = tf.data.Dataset.from_generator(
    sequence_generator,
    output_signature=(
        tf.TensorSpec(shape=(max_sequence_len - 1,), dtype=tf.int32),
        tf.TensorSpec(shape=(total_words,), dtype=tf.float32),
    ),
)
dataset = (
    dataset.shuffle(buffer_size=10000).batch(64).prefetch(tf.data.AUTOTUNE)
)
# save model for later use if needed
checckpoint_path = "training_1/model.keras"
checkpoint_dir = os.path.dirname(checckpoint_path)

steps_per_epoch = math.ceil(total_sequences / 64)
model_final = build_model(2, 256)  # 2 layers, 256 nodes
print("Final model")
model_final.fit(
    dataset,
    epochs=100,
    steps_per_epoch=steps_per_epoch,
    verbose=1,
    callbacks=[csv_logger],
)

model_path = "training_1/model.keras"

loaded_model = tf.keras.models.load_model(model_path)

text = "The research shows"
response = generate_chatbot_response(text, 65, loaded_model)
with open("model_output", "w") as f:
    print(response, file=f)


###############
# OLD MODELS  #
###############

# 1 layer, 50 nodes, 10 epochs
# model_1_x = build_model(1, 50)
# print('Model 1 -- 10 epochs')
# model_1_10 = model_1_x
# model_1_10.fit(X, y, epochs=10, verbose=1)

# 5 layers, 50 nodes, 10 epochs
# model_5_10 = build_model(5, 50)
# print('Model 2 -- 10 epochs')
# model_5_10.fit(X, y, epochs=10, verbose=1)

# 5 Layers, 100 Nodes, 10 epochs
# model_5_100 = build_model(5, 100)
# print('Model 3 -- 10 epochs')
# model_5_100.fit(X, y, epochs=10, verbose=1, callbacks=[csv_logger])

# 5 Layers, 100 Nodes, 50 epochs
# model_5_100_50 = build_model(5, 100)
# print('Model 4 -- 50 epochs')
# model_5_100_50.fit(X, y, epochs=50, verbose=1, callbacks=[csv_logger])
