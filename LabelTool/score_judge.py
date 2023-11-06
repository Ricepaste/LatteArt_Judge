import os
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox
# import csv
import numpy as np
import pandas as pd
import math
import shutil
import random

IMAGE_SIZE = 300
WINDOW_SIZE = 500
USER_INPUT_BAR_SIZE = 10
STOP = 0
i = 0
CHECK_FLAG = 0


def import_image(path):
    global img
    img = Image.open(path)
    resize_image = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    tk_img = ImageTk.PhotoImage(resize_image)
    lab = tk.Label(window, image=tk_img)
    lab.image = tk_img  # 保留對圖像物件的引用以避免被垃圾回收
    return lab


def score_judge():
    global curr_img, arr, min_index, STOP, i
    try:
        arr = pd.read_csv('./LabelTool/Score.csv', header=None)
        arr = np.array(arr.values)
    except pd.errors.EmptyDataError:
        print("Empty")
        arr = np.array([])
        
    arr = arr.tolist()
     
    min_value = 100
    min_index = 0
    
    # temp_length = []

    # if (len(arr) != len(result)):
    #     min_index = len(arr)
    # else:
    #     if (STOP == len(result)):
    #         window_close()
    #     for i in range(len(arr)):
    #         length = 0
    #         for j in range(len(arr[i])):
    #             try:
    #                 if (math.isnan(arr[i][j])):
    #                     continue
    #             except:
    #                 if (arr[i][j] == 'N'):
    #                     continue
    #                 else:
    #                     length += 1
    #         if (length < min_value):
    #             min_value = length
    #             min_index = i
    #         temp_length.append(length)
                
    # min_value = min(temp_length)
    # min_index = temp_length.index(min_value)
    # STOP += 1

    
    temp_length = []
            
    if (len(arr) != len(result)):
        min_index = len(arr)
    else:
        if (STOP == len(result)):
            window_close()
            # exit(0)
        for k in range(len(arr)):
            length = 0
            for p in range(len(arr[k])):
                # if (math.isnan(arr[k][p])):
                #     length += 1
                if (math.isnan(arr[k][p])):
                    continue
                else:
                    length += 1
            temp_length.append(length)
                    
        min_value = min(temp_length)
        min_index = temp_length.index(min_value)
    STOP += 1

    try:
        curr_img = import_image(result[min_index])
        curr_image_index = tk.Label(window, text = "{}/{}".format(min_index, len(result)))


        curr_img.pack()

        button.place(x=WINDOW_SIZE/2+USER_INPUT_BAR_SIZE *
                     7/2+15, y=IMAGE_SIZE+45)

        user_input.place(x=WINDOW_SIZE/2-USER_INPUT_BAR_SIZE *
                         7/2, y=IMAGE_SIZE+50)

        input_title.place(x=WINDOW_SIZE/2-USER_INPUT_BAR_SIZE *
                          7/2-70, y=IMAGE_SIZE+47)

        input_remind.place(x=WINDOW_SIZE/2-USER_INPUT_BAR_SIZE *
                           7/2-20, y=IMAGE_SIZE+80)
        
        curr_image_index.place(x=WINDOW_SIZE - 50, y=0
                               )
    except:
        pass

def window_close():
    window.destroy()


def send_score(event=None):
    # send score to excel to the corresponding index
    global i, STOP
    curr_img.destroy()
    get_num_from_bar()
    if (not (score.isdigit()) or (int(score) < 0 or int(score) > 10)):
        messagebox.showerror(title="錯誤輸入", message="媽的文盲")
        # i-=1
        STOP-=1
    else:
        write_score(min_index, score)
        # print("CHECK FLAG2: ", CHECK_FLAG)
    clearBar()
    score_judge()

def get_num_from_bar():
    global score
    score = user_input.get()
    # print(score)


def clearBar():
    user_input.delete(0, 'end')


def write_score(index, score):
    global CHECK_FLAG
    temp_list = []
    try:
        temp_list = arr[index]
        for k in range(len(temp_list)):
            if (CHECK_FLAG == 0 and math.isnan(temp_list[k])):
                temp_list.append(score)
                break
            else:
                if (CHECK_FLAG == 1 and math.isnan(temp_list[k]) and k == len(temp_list)-1):
                    temp_list[-1] = score
                else:
                    if (k == len(temp_list)-1):
                        temp_list.append(score)
                        break
                    else:
                        continue
                

        arr[index] = temp_list
    except:
        temp_list.append(score)
        try:
            arr[index] = temp_list
        except:
            arr.append(temp_list)
    # print(arr)

    CHECK_FLAG = 1
    # print("CHECK FLAG: ", CHECK_FLAG)
    # Convert arr to a DataFrame
    df = pd.DataFrame(arr)
    # print(df)

    # Save df into Score.csv
    df.to_csv('./LabelTool/Score.csv', index=False, header=False)
    # files = os.listdir('./LabelTool/')
    # i = 0
    name = 'BackUp.csv'
    # while name in files:
    #     i += 1
    #     name = 'BackUp' + str(i) + '.csv'

    df.to_csv('./LabelTool/' + name, index=False, header=False)


dirpath = r"./LabelTool/backup27"
result = [os.path.join(dirpath, f) for f in os.listdir(
    dirpath) if os.path.isfile(os.path.join(dirpath, f))]

window = tk.Tk()
window.title('Score')
window.geometry("{}x{}".format(WINDOW_SIZE, WINDOW_SIZE))

button = tk.Button(window, text="Send", command=send_score)
user_input = tk.Entry(window, width=USER_INPUT_BAR_SIZE)
input_title = tk.Label(window, text="拉花評分: ", font=("Arial", 10))
input_remind = tk.Label(window, text="(請輸入介於0~10分的整數)", font=("Arial", 10))

score_judge()

window.bind('<Return>', send_score)
window.mainloop()


# 11/6 補nan
nan_data = pd.read_csv('./LabelTool/Score.csv', header=None)
nan_data = np.array(nan_data.values)
nan_data = nan_data.tolist()
lengthofdata = max([len(nan_data[i]) for i in range(len(nan_data))])
# print(data)
if (len(nan_data) != len(result)):
    # insert nan into data
    for i in range(len(result)-len(nan_data)):
        nan_data.append([math.nan])
for k in range(len(nan_data)):
    if (len(nan_data[k]) != lengthofdata):
        for p in range(lengthofdata-len(nan_data[k])):
            nan_data[k].append(math.nan)
        
df = pd.DataFrame(nan_data)
df.to_csv('./LabelTool/Score.csv', index=False, header=False)

# 11/4

Train_Size = 0.7
Test_Size = 0.3

# Read the data
data = pd.read_csv('./LabelTool/Score.csv', header=None)
data = np.array(data.values)

average = []

for i in range(len(data)):
    sum = 0
    count = 0
    for j in range(len(data[i])):
        if (data[i][j] == "N"):
            continue
        # if (math.isnan(data[i][j])):
        #     break
        else:
            sum += data[i][j]
            count += 1
    try:
        average.append(sum/count)
    except:
        pass

# print(average)

cut = int(len(average)*Train_Size)

# train = average[:cut]
# test = average[cut:]

# train_data = result[:cut]
# test_data = result[cut:]

average_index_shuffle = [i for i in range(len(average))]
random.shuffle(average_index_shuffle)
train_index_shuffle = average_index_shuffle[:cut]
test_index_shuffle = average_index_shuffle[cut:]

dir1 = os.listdir('./LabelTool/train/images')
dir2 = os.listdir('./LabelTool/val/images')
dir3 = os.listdir('./LabelTool/train/labels')
dir4 = os.listdir('./LabelTool/val/labels')

try:
    if (len(dir1) != 0 or len(dir2) != 0 or len(dir3) != 0 or len(dir4) != 0):
        raise Exception("The folder is not empty")
except:
    for file in os.listdir('./LabelTool/train/images'):
        os.remove('./LabelTool/train/images/' + file)
    for file in os.listdir('./LabelTool/val/images'):
        os.remove('./LabelTool/val/images/' + file)
    for file in os.listdir('./LabelTool/train/labels'):
        os.remove('./LabelTool/train/labels/' + file)
    for file in os.listdir('./LabelTool/val/labels'):
        os.remove('./LabelTool/val/labels/' + file)

# copy the image to the corresponding folder
for m in train_index_shuffle:
    shutil.copy(result[m], './LabelTool/train/images')
    temp = result[m].split('./LabelTool/backup27')
    fuck = temp[1].split('.jpg')
    file_path = '.\\LabelTool\\train\\labels{}.txt'.format(fuck[0])
    content = average[m]
    with open(file_path, 'w+') as f:
        f.write(str(content))

# index_temp = 02

for n in test_index_shuffle:
    temp = result[n].split('./LabelTool/backup27')
    fuck = temp[1].split('.jpg')
    shutil.copy(result[n], './LabelTool/val/images')
    file_path = '.\\LabelTool\\val\\labels{}.txt'.format(fuck[0])
    content = average[n]
    # index_temp += 1
    with open(file_path, 'w+') as f:
        f.write(str(content))