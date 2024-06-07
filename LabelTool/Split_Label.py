import pandas as pd
import numpy as np

"""
!! Caution: 多個圖片會有相同分數，但切了十等分後會有不同的label
"""

class Split_Label:
    def __init__(self):
        self.df = pd.read_csv("./LabelTool/ForTestingImage.csv", sep=",")
        self.split()
        
    def split(self):
        # 根據score進行排序
        self.df = self.df.sort_values(by="Score", ascending=False)
        
        # 計算每個標籤的分界點
        n = len(self.df)
        bin_edges = np.linspace(0, n, 11)  # 10 個分界點，11 個邊界

        self.df['Label'] = 0
        
        # 根據分界點分配 label
        for i in range(10):
            start = int(bin_edges[i])
            end = int(bin_edges[i + 1])
            self.df.iloc[start:end, self.df.columns.get_loc('Label')] = 10 - i

        self.df = self.df.drop(columns=['Score'])
        self.df.to_csv("./LabelTool/Image_label.csv", index=False)