import pandas as pd

# GCMD
gcmd_textblob = pd.read_csv(
    "/Users/shelbybagwell/Documents/UTK-MSCS/coursework/Spring2026/NLP_COSC-524/chatbot_project/mod5/results/gcmd_science_keywords_textblob_sentiment.csv"
)
gcmd_LVC = pd.read_csv(
    "/Users/shelbybagwell/Documents/UTK-MSCS/coursework/Spring2026/NLP_COSC-524/chatbot_project/mod5/results/gcmd_science_keywords_LVC_sentiment.csv"
)
gcmd_lab12 = pd.read_csv(
    "/Users/shelbybagwell/Documents/UTK-MSCS/coursework/Spring2026/NLP_COSC-524/chatbot_project/mod5/results/gcmd_sentiment_analysis_results.csv"
)

random_gcmd_textblob = gcmd_textblob.sample(10, random_state=42)
random_gcmd_LVC = gcmd_LVC.sample(10, random_state=42)
random_gcmd_lab12 = gcmd_lab12.sample(10, random_state=42)

dfs = [random_gcmd_textblob, random_gcmd_LVC, random_gcmd_lab12]
labels = ["TextBlob", "LVC", "Lab12"]
combined_df = (
    pd.concat(dfs, keys=labels, names=["Source_label"])
    .reset_index(level=0)
    .reset_index(drop=True)
)
combined_df.to_csv("gcmd_combined_sentiment.csv", index=False)

# abstracts
abstracts_textblob = pd.read_csv(
    "/Users/shelbybagwell/Documents/UTK-MSCS/coursework/Spring2026/NLP_COSC-524/chatbot_project/mod5/results/abstracts_textblob_sentiment.csv"
)
abstracts_LVC = pd.read_csv(
    "/Users/shelbybagwell/Documents/UTK-MSCS/coursework/Spring2026/NLP_COSC-524/chatbot_project/mod5/results/abstracts_LVC_sentiment.csv"
)
abstracts_lab12 = pd.read_csv(
    "/Users/shelbybagwell/Documents/UTK-MSCS/coursework/Spring2026/NLP_COSC-524/chatbot_project/mod5/results/abstracts_sentiment_analysis_results.csv"
)

random_abstracts_textblob = abstracts_textblob.sample(10, random_state=42)
random_abstracts_LVC = abstracts_LVC.sample(10, random_state=42)
random_abstracts_lab12 = abstracts_lab12.sample(10, random_state=42)

dfs = [random_abstracts_textblob, random_abstracts_LVC, random_abstracts_lab12]
labels = ["TextBlob", "LVC", "Lab12"]
combined_df = (
    pd.concat(dfs, keys=labels, names=["Source_label"])
    .reset_index(level=0)
    .reset_index(drop=True)
)
combined_df.to_csv("abstracts_combined_sentiment.csv", index=False)

# chain of thought
cot_textblob = pd.read_csv(
    "/Users/shelbybagwell/Documents/UTK-MSCS/coursework/Spring2026/NLP_COSC-524/chatbot_project/mod5/results/chain_of_thought_textblob_sentiment.csv"
)
cot_LVC = pd.read_csv(
    "/Users/shelbybagwell/Documents/UTK-MSCS/coursework/Spring2026/NLP_COSC-524/chatbot_project/mod5/results/chain_of_thought_LVC_sentiment.csv"
)
cot_lab12 = pd.read_csv(
    "/Users/shelbybagwell/Documents/UTK-MSCS/coursework/Spring2026/NLP_COSC-524/chatbot_project/mod5/results/cot_sentiment_analysis_results.csv"
)
random_cot_textblob = cot_textblob.sample(10, random_state=42)
random_cot_LVC = cot_LVC.sample(10, random_state=42)
random_cot_lab12 = cot_lab12.sample(10, random_state=42)
dfs = [random_cot_textblob, random_cot_LVC, random_cot_lab12]
labels = ["TextBlob", "LVC", "Lab12"]
combined_df = (
    pd.concat(dfs, keys=labels, names=["Source_label"])
    .reset_index(level=0)
    .reset_index(drop=True)
)
combined_df.to_csv(
    "mod5/results/chain_of_thought_combined_sentiment.csv", index=False
)
