import spacy
import nltk
from nltk import word_tokenize, pos_tag
from spacy.tokens import Doc
from spacy import displacy

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nlp = spacy.load("en_core_web_sm")

OUTPUT_DIRECTORY = "training_output/"


def read_file(filepath: str) -> str:
    print(f"Reading file {filepath}...")
    with open(filepath, "r") as f:
        text = f.read()
    return text


def get_tokens(text: str, data_source: str) -> list[str]:
    print(f"Getting tokens for {data_source}...")
    text = text.lower()
    return word_tokenize(text)


# POS tagging
def get_pos(text: str, data_source: str):
    tokens = get_tokens(text, data_source)
    print(f"Getting POS tags for {data_source}...")
    pos_tags = pos_tag(tokens)

    output_file = f"{OUTPUT_DIRECTORY}{data_source}/pos.txt"
    with open(output_file, "w+") as f:
        for tag in pos_tags:
            f.write(f"{tag}\n")


# NER
def get_NER(text: str, data_source: str):
    print(f"Getting NER for {data_source}...")
    doc = nlp(text)

    output_file = f"{OUTPUT_DIRECTORY}{data_source}/ner.txt"
    with open(output_file, "w+") as f:
        for ent in doc.ents:
            f.write(f"{ent.text} {ent.label_}\n")


# dependency parsing
def parse_dependencies(text: str, data_source: str) -> Doc:
    print(f"Parsing Dependencies for {data_source}...")
    doc = nlp(text)

    output_file = f"{OUTPUT_DIRECTORY}{data_source}/dependency.txt"
    with open(output_file, "w+") as f:
        for token in doc:
            f.write(f"{token.text} ({token.dep_}): {token.head.text}\n")
    return doc


def visualize_dependencies(doc: Doc):
    displacy.serve(doc, style="dep", port=5001)


def for_lab(filepath: str, data_source: str):
    with open(filepath, "r") as f:
        lines = f.readlines()

    sample = lines[0]

    output_dir = "training_output/for_lab/"
    tokens = word_tokenize(sample)
    pos_tags = pos_tag(tokens)
    doc = nlp(sample)
    with open(f"{output_dir}sample_outputs.txt", "a") as f:
        f.write(f"{data_source}\n")
        f.write(f"{sample}\n")
        f.write("POS Tags:\n")
        for tag in pos_tags:
            f.write(f"{tag}")
        f.write("\n\nNER:\n")
        for ent in doc.ents:
            f.write(f"{ent.text} {ent.label_}\n")
        f.write("\n\nDependency Parsing:\n")
        for token in doc:
            f.write(f"{token.text} ({token.dep_}): {token.head.text}\n")
        f.write("-" * 20 + "\n\n")


# main function calls

for_lab("training_data/cleaned_data/gcmd_science_keywords_cleaned.txt", "gcmd")
for_lab("training_data/cleaned_data/abstracts_cleaned.txt", "abstracts")
for_lab(
    "training_data/cleaned_data/chain_of_thought_cleaned.txt",
    "chain_of_thought",
)

# GCMD keywords
# gcmd_text = read_file(
#     "training_data/cleaned_data/gcmd_science_keywords_cleaned.txt"
# )
# get_pos(gcmd_text, "gcmd")
# get_NER(gcmd_text, "gcmd")
# gcmd_doc = parse_dependencies(gcmd_text, "gcmd")

# # Abstracts
# abstracts_text = read_file("training_data/cleaned_data/abstracts_cleaned.txt")
# get_pos(abstracts_text, "abstracts")
# get_NER(abstracts_text, "abstracts")
# abstracts_doc = parse_dependencies(gcmd_text, "abstracts")

# # Chain of thought
# cot_text = read_file("training_data/cleaned_data/chain_of_thought_cleaned.txt")
# get_pos(cot_text, "chain_of_thought")
# get_NER(cot_text, "chain_of_thought")
# cot_doc = parse_dependencies(cot_text, "chain_of_thought")
