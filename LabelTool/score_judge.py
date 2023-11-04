import os
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox
import csv
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
    # print(arr)
    arr = arr.tolist()
    
    min_value = 100
    min_index = 0
    

    while(1):
        length = 0
        if (STOP == len(result)):
            window_close()
            # exit(0)
        try:
            length = len(arr[i])
            if (min_value > length):
                min_value = length
                min_index = i
                STOP += 1
                i += 1
                break
            else:
                if (i == len(result)-1):
                    window_close()
                    # exit(0)
        except:
            min_index = i
            STOP += 1
            i += 1
            break
          
    try:
        curr_img = import_image(result[min_index])
    
        curr_img.pack()

        button.place(x=WINDOW_SIZE/2+USER_INPUT_BAR_SIZE*7/2+15, y=IMAGE_SIZE+45)

        user_input.place(x=WINDOW_SIZE/2-USER_INPUT_BAR_SIZE*7/2, y=IMAGE_SIZE+50)

        input_title.place(x=WINDOW_SIZE/2-USER_INPUT_BAR_SIZE *
                        7/2-70, y=IMAGE_SIZE+47)

        input_remind.place(x=WINDOW_SIZE/2-USER_INPUT_BAR_SIZE *
                        7/2-20, y=IMAGE_SIZE+80)
    except:
        pass


def window_close():
    window.destroy()


def send_score():
    # send score to excel to the corresponding index
    curr_img.destroy()
    get_num_from_bar()
    if (not(score.isdigit()) or (int(score) < 0 or int(score) > 10)):
        messagebox.showerror(title="錯誤輸入", message="媽的文盲")
    else: 
        write_score(min_index, score)  
    clearBar()
    score_judge()

def get_num_from_bar():
    global score
    score = user_input.get()
    # print(score)

def clearBar():
    user_input.delete(0, 'end')

def write_score(index, score):
    temp_list = []
    try:
        temp_list = arr[index]
        for k in range(len(temp_list)):
            if (math.isnan(temp_list[k])):
                temp_list[k] = score
                break
            else:
                if (k == len(temp_list)-1):
                    temp_list.append(score)
                    break
                else:
                    continue

        arr[index] = temp_list
    except:
        temp_list.append(score)
        arr.append(temp_list)
    # print(arr)

    # Convert arr to a DataFrame
    df = pd.DataFrame(arr)
    # print(df)

    # Save df into Score.csv
    df.to_csv('./LabelTool/Score.csv', index=False, header=False)
    files = os.listdir('./LabelTool/')
    i = 0
    name = 'BackUp.csv'
    for file in files:
        if file == name:
            i += 1
            name = 'BackUp' + str(i) + '.csv'
        else:
            df.to_csv('./LabelTool/' + name, index=False, header=False)
            break
    
    
    

        

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

window.mainloop()


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
        if (math.isnan(data[i][j])):
            break
        else:
            sum += data[i][j]
            count += 1
    average.append(sum/count)
    
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

dir1 = os.listdir('./LabelTool/Train/Image') 
dir2 = os.listdir('./LabelTool/Test/Image')
dir3 = os.listdir('./LabelTool/Train/Label')
dir4 = os.listdir('./LabelTool/Test/Label')

try:
    if (len(dir1) != 0 or len(dir2) != 0 or len(dir3) != 0 or len(dir4) != 0):
        raise Exception("The folder is not empty")
except:
    for file in os.listdir('./LabelTool/Train/Image'):
        os.remove('./LabelTool/Train/Image/' + file)
    for file in os.listdir('./LabelTool/Test/Image'):
        os.remove('./LabelTool/Test/Image/' + file)
    for file in os.listdir('./LabelTool/Train/Label'):
        os.remove('./LabelTool/Train/Label/' + file)
    for file in os.listdir('./LabelTool/Test/Label'):
        os.remove('./LabelTool/Test/Label/' + file)

# copy the image to the corresponding folder
for m in train_index_shuffle:
    shutil.copy(result[m], './LabelTool/Train/Image')
    temp = result[m].split('./LabelTool/backup27')
    file_path = '.\\LabelTool\\Train\\Label{}.txt'.format(temp[1])
    content = average[m]
    with open(file_path, 'w') as f:
        f.write(str(content))

# index_temp = 0

for n in test_index_shuffle:
    temp = result[n].split('./LabelTool/backup27')
    shutil.copy(result[n], './LabelTool/Test/Image')
    file_path = '.\\LabelTool\\Test\\Label{}.txt'.format(temp[1])
    content = average[n]
    # index_temp += 1
    with open(file_path, 'w') as f:
        f.write(str(content))