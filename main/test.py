import pandas as pd
import numpy as np

arr = pd.read_csv('./LabelTool/Pro.csv', header=None)
arr = np.array(arr.values).flatten().tolist()
print(arr)
print(np.round(arr))
