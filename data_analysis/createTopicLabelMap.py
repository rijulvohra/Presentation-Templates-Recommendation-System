import pandas as pd
import json

if __name__ == "__main__":
    input_path = "../data/topic_classification_data/test.csv"
    df = pd.read_csv(input_path)
    topic_label_map = {}
    for k, v in df.iterrows():
        topic = v["topic"]
        topic_label = v["topic_label"]

        if topic_label_map.get(topic_label) is None:
            topic_label_map[topic_label] = topic.split("-")[-1]


    out_path = "topic_label_map.json"
    with open(out_path, "w") as f:
        json.dump(topic_label_map, f)

