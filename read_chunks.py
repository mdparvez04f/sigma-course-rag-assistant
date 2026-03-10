import os
import json
import pandas as pd

json_folder = "jsons"
records = []
chunk_id = 0

for file in os.listdir(json_folder):
    if file.endswith(".json"):
        with open(os.path.join(json_folder, file)) as f:
            data = json.load(f)

        for chunk in data["chunks"]:
            chunk["chunk_id"] = chunk_id
            records.append(chunk)
            chunk_id += 1

df = pd.DataFrame(records)

print(df.head())