import os
import pandas as pd

class Preprocessing:
    def __init__(self, folder_name, reset):
        self.reset = reset
        self.folder_name = folder_name
        self.reset_csv(self.reset)
        temp = self.rename(self.get_file_list())
        self.create_csv(temp)

    def get_file_list(self):
        file_list = os.listdir(f"./LabelTool/{self.folder_name}")
        file_list = [file for file in file_list if not file.endswith(".csv")]
        return file_list

    def rename(self, file_list):
        for i in range(len(file_list)):
            try:
                if os.path.exists(f"./LabelTool/{self.folder_name}/" + str(i) + ".jpg"):
                    continue
                else:
                    os.rename(f"./LabelTool/{self.folder_name}/" + file_list[i], f"./LabelTool/{self.folder_name}/" + str(i) + ".jpg")
            except:
                continue
        return file_list
            
    def create_csv(self, file_list):
        if not os.path.exists(f"./LabelTool/{self.folder_name}/Score.csv"):
            # 紀錄圖片與分數
            with open(f"./LabelTool/{self.folder_name}/Score.csv", "w") as f:
                f.write("ImageID,Score\n")
                for i in range(len(file_list)):
                    f.write(str(i) + ",1500\n")
        else:
            df = pd.read_csv(f"./LabelTool/{self.folder_name}/Score.csv", sep=",")
            if len(df) != len(file_list):
                for i in range(len(file_list)):
                    if i not in df["ImageID"].values.tolist():
                        df = pd.concat([df, pd.DataFrame({"ImageID":[str(i)], "Score":[1500]})], ignore_index=True)
                df.to_csv(f"./LabelTool/{self.folder_name}/Score.csv", index=False)
                
        # 紀錄使用者喜好
        if not os.path.exists(f"./LabelTool/{self.folder_name}/record.csv"):
            record = pd.DataFrame(columns=["WinnerID", "LoserID"])
            record.to_csv(f"./LabelTool/{self.folder_name}/record.csv", index=False)
            
    def reset_csv(self, RESET):
        if RESET == True:
            record = pd.DataFrame(columns=["WinnerID", "LoserID"])
            record.to_csv(f"./LabelTool/{self.folder_name}/record.csv", index=False)
            self.create_csv(self.get_file_list())