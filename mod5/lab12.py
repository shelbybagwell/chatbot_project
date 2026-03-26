#! /usr/bin/python3.10


from sklearn.calibration import LinearSVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.layers import Embedding
from sklearn.feature_extraction.text import CountVectorizer


from sklearn.model_selection import train_test_split


# ***** additional imports needed for this lab *****
import sys, getopt  #   used to read command line arguments
from nltk.tokenize import RegexpTokenizer

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import numpy as np  # linear algebra
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# **************************************************


def main():

    # BLOCK 2 (CORE SOLUTION)
    # We now read the file provided by the user into a corpus
    inputfile = "/Users/shelbybagwell/Documents/UTK-MSCS/coursework/Spring2026/NLP_COSC-524/chatbot_project/training_data/cleaned_data/chain_of_thought_cleaned.txt"

    tokenizer = RegexpTokenizer(r"\w+")

    # Read in the data and strip newlines from the end
    corpus = [line.strip() for line in open(inputfile, "r")]
    corpus[:] = [tokenizer.tokenize(x) for x in corpus if x != ""]
    corpus[:] = [" ".join(x) for x in corpus]

    # Load data

    df = pd.read_csv(
        "/Users/shelbybagwell/Documents/UTK-MSCS/coursework/Spring2026/NLP_COSC-524/chatbot_project/training_data/IMDB_Dataset.csv"
    )

    X = df["review"]
    y = df["sentiment"].map({"positive": 1, "negative": 0})

    # Split into training and testing data

    # Assume X is your list of texts and y is the corresponding sentiments
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Prepare tokenizer
    t = Tokenizer(num_words=20000)
    t.fit_on_texts(X_train)

    # Convert text into sequences of integers
    sequences = t.texts_to_sequences(X_train)
    test_sequences = t.texts_to_sequences(X_test)

    # Pad the sequences so they are all the same length
    train_padded = pad_sequences(sequences, maxlen=100)
    test_padded = pad_sequences(test_sequences, maxlen=100)

    # Define the model
    model = Sequential()
    model.add(Embedding(20000, 100, input_length=100))
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 5, activation="relu"))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(100))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    # Train the model
    model.fit(train_padded, np.array(y_train), validation_split=0.4, epochs=3)

    # Evaluate on test set
    loss, accuracy = model.evaluate(test_padded, np.array(y_test))
    print("Test Accuracy: %f" % (accuracy * 100))

    print(f"Predicting on {inputfile}")
    input_sequences = t.texts_to_sequences(corpus)
    input_padded = pad_sequences(input_sequences, maxlen=100)
    predictions = model.predict(input_padded)

    results = []
    for i, prob in enumerate(predictions):
        score = prob[0]
        label = "positive" if score >= 0.5 else "negative"
        results.append(
            {
                "Text": corpus[i],
                "Predicted_Sentiment": label,
                "Raw_Score": round(score, 4),
            }
        )

    outputdf = pd.DataFrame(results)
    outputdf.to_csv("sentiment_analysis_results.csv", index=False)


if __name__ == "__main__":
    main()
