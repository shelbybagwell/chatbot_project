from textblob import TextBlob
import pandas as pd
from sklearn import svm
from sklearn.calibration import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def read_file(filepath: str) -> list[str]:
    with open(filepath, "r") as f:
        text = []
        for line in f:
            text.append(line.strip())
    return text


def analyze_sentiment(text: list[str]) -> pd.DataFrame:
    results = []
    for line in text:
        blob = TextBlob(line)
        sentiment = blob.sentiment
        results.append(
            {
                "text": line,
                "polarity": sentiment.polarity,
                "subjectivity": sentiment.subjectivity,
            }
        )
    df = pd.DataFrame(results)
    return df


def LVC_training(text: list[str]) -> pd.DataFrame:
    df = pd.read_csv("training_data/IMDB_Dataset.csv")

    X = df["review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vector = TfidfVectorizer()

    X_train_vectorized = vector.fit_transform(X_train)
    X_test_vectorized = vector.transform(X_test)

    # model
    model = svm.LinearSVC()
    model.fit(X_train_vectorized, y_train)

    sample_vec = vector.transform(text)
    prediction = model.predict(sample_vec)

    results = []
    for line, pred in zip(text, prediction):
        results.append(
            {
                "text": line,
                "predicted_sentiment": pred,
            }
        )

    dfresults = pd.DataFrame(results)
    return dfresults


if __name__ == "__main__":
    f1 = "training_data/cleaned_data/gcmd_science_keywords_cleaned.txt"
    f1_data = read_file(f1)
    df_1 = analyze_sentiment(f1_data)
    df_1.to_csv("gcmd_science_keywords_textblob_sentiment.csv", index=False)
    df_2 = LVC_training(f1_data)
    df_2.to_csv("gcmd_science_keywords_LVC_sentiment.csv", index=False)

    f2 = "training_data/cleaned_data/chain_of_thought_cleaned.txt"
    f2_data = read_file(f2)
    df_1 = analyze_sentiment(f2_data)
    df_1.to_csv("chain_of_thought_textblob_sentiment.csv", index=False)
    df_2 = LVC_training(f2_data)
    df_2.to_csv("chain_of_thought_LVC_sentiment.csv", index=False)

    f3 = "training_data/cleaned_data/abstracts_cleaned.txt"
    f3_data = read_file(f3)
    df_1 = analyze_sentiment(f3_data)
    df_1.to_csv("abstracts_textblob_sentiment.csv", index=False)
    df_2 = LVC_training(f3_data)
    df_2.to_csv("abstracts_LVC_sentiment.csv", index=False)
