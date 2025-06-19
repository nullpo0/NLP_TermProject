import pandas as pd

a = pd.read_json("./data/test_cls.json")

for i in a['question']:
    print(i)