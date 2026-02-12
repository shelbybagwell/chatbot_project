import requests
import json

base_url = "https://api.semanticscholar.org/graph/v1/paper/"
search_query_params = "atmospheric+science+earth+science"
data_type_param = "fields=url,abstract,authors"

filepath = "/Users/shelbybagwell/Documents/UTK-MSCS/coursework/Spring2026/NLP_COSC-524/chatbot_project/training_data/semantic_scholar_raw_results.txt"
json_filepath = "/Users/shelbybagwell/Documents/UTK-MSCS/coursework/Spring2026/NLP_COSC-524/chatbot_project/training_data/semantic_scholar_json.json"

r = requests.get(
    f"{base_url}search/bulk?query={search_query_params}&{data_type_param}"
)

if r.status_code == 200:
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(r.text)

    try:
        data = r.json()
        with open(json_filepath, "w", encoding="utf-8") as jf:
            json.dump(data, jf, indent=4)
    except requests.JSONDecodeError:
        print("Response was not in JSON format, raw text still provided")
else:
    print(
        f"Something went wrong. Request returned with error code {r.status_code}"
    )
