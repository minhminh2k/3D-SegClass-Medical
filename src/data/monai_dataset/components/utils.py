import json
import os

def datafold_read(
    data_dir: str, 
    datalist_json: str, 
    fold: int = 0, 
    key="training"
):
    with open(datalist_json) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(data_dir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(data_dir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val
