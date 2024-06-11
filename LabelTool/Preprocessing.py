import os
import pandas as pd

def get_file_list():
    file_list = os.listdir("./LabelTool/backup27")
    return file_list

def rename(file_list):
    for i in range(len(file_list)):
        try:
            os.rename("./LabelTool/backup27/" + file_list[i], "./LabelTool/backup27/" + str(i) + ".jpg")
        except:
            continue
    return file_list
        
def create_csv(file_list):
    if not os.path.exists("./LabelTool/Score.csv"):
        # 紀錄圖片與分數
        with open("./LabelTool/Score.csv", "w") as f:
            f.write("ImageID,Score\n")
            for i in range(len(file_list)):
                f.write(str(i) + ",1500\n")
    else:
        df = pd.read_csv("./LabelTool/Score.csv", sep=",")
        if len(df) != len(file_list):
            for i in range(len(file_list)):
                if i not in df["ImageID"].values.tolist():
                    df = pd.concat([df, pd.DataFrame({"ImageID":[str(i)], "Score":[1500]})], ignore_index=True)
            df.to_csv("./LabelTool/Score.csv", index=False)
            
    # 紀錄使用者喜好
    if not os.path.exists("./LabelTool/record.csv"):
        with open("./LabelTool/record.csv", "w") as f:
            f.write("WinnerID, LoserID\n")
            
            
temp = rename(get_file_list())
create_csv(temp)
                
                
        
