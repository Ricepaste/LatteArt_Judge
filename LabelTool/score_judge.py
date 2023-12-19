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
BACK_FLAG = 0
temp_index = []
img_path = []


def import_image(path):
    global img
    img = Image.open(path)
    resize_image = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    tk_img = ImageTk.PhotoImage(resize_image)
    lab = tk.Label(window, image=tk_img)
    lab.image = tk_img  # 保留對圖像物件的引用以避免被垃圾回收
    if (path in img_path):
        pass
    else:
        img_path.append(path)
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
                if (arr[k][p] == "."):
                    length += 1
                    continue
                elif (math.isnan(float(arr[k][p]))):
                    continue
                else:
                    length += 1
            temp_length.append(length)

        min_value = min(temp_length)
        min_index = temp_length.index(min_value)
        # save the index of the image that has been scored
    temp_index.append(min_index)
    STOP += 1

    try:
        curr_img = import_image(result[min_index])
        curr_image_index = tk.Label(
            window, text="{}/{}".format(min_index, len(result)))

        curr_img.pack()

        button.place(x=WINDOW_SIZE/2+USER_INPUT_BAR_SIZE *
                     7/2+15, y=IMAGE_SIZE+45)

        user_input.place(x=WINDOW_SIZE/2-USER_INPUT_BAR_SIZE *
                         7/2, y=IMAGE_SIZE+50)

        input_title.place(x=WINDOW_SIZE/2-USER_INPUT_BAR_SIZE *
                          7/2-70, y=IMAGE_SIZE+47)

        input_remind.place(x=WINDOW_SIZE/2-USER_INPUT_BAR_SIZE *
                           7/2-90, y=IMAGE_SIZE+80)

        curr_image_index.place(x=WINDOW_SIZE - 50, y=0
                               )
        back_button.place(x=WINDOW_SIZE/2+USER_INPUT_BAR_SIZE *
                          7/2+60, y=IMAGE_SIZE+45)
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
        try:
            if (int(score) == -1):
                write_score(min_index, ".")
        except:
            messagebox.showerror(title="錯誤輸入", message="媽的文盲")
            # i-=1
            STOP -= 1
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


"""
def back_menu():
    global img_path
    temp_window = tk.Tk()
    temp_window.title('Back Menu')
    temp_window.geometry("{}x{}".format(WINDOW_SIZE, WINDOW_SIZE))
    
    frame = tk.Frame(temp_window, height = 50, width = 75)
    temp_scrollbar = tk.Scrollbar(frame)
    temp_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    # put image on the frame
    for i in range(len(img_path)):
        temp_img = Image.open(img_path[i])
        resize_image = temp_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(resize_image)
        lab = tk.Label(frame, image=tk_img)
        lab.image = tk_img
        lab.pack()
        print(img_path[i])
        
    # text = tk.Text(frame, height = 30, width = 60, yscrollcommand = temp_scrollbar.set)
    # text.pack()
    
    # temp_scrollbar.config(command = text.yview)
    frame.pack()
    temp_window.mainloop()
"""


def back_menu():
    global img_path, temp_index, temp_window
    temp_window = tk.Toplevel()  # Use Toplevel instead of Tk
    temp_window.title('Back Menu')
    temp_window.geometry("{}x{}".format(WINDOW_SIZE, WINDOW_SIZE))

    frame = tk.Frame(temp_window, width=200, height=175)
    frame.pack()

    # text = tk.Text(frame, text="")

    # put image on the canvas

    canvas = tk.Canvas(frame, width=200, height=350,
                       scrollregion=(0, 0, 200, 175*len(img_path)))
    scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # temp_img_index = tk.Label(temp_window, text="Image index: ")

    temp_entry_label = tk.Label(temp_window, text="Change score: ")
    temp_entry_label.place(x=WINDOW_SIZE/2-100, y=WINDOW_SIZE/2+142.5)
    temp_entry = tk.Entry(temp_window, width=15)
    temp_entry.place(x=WINDOW_SIZE/2-10, y=WINDOW_SIZE/2+145)
    
    # temp_button = tk.Button(temp_window, text="Change", command=write_score(temp_index[img_path.index(image_path)], temp_entry.get()))
    temp_button.place(x=WINDOW_SIZE/2+100, y=WINDOW_SIZE/2+162.5)
    
    for i in range(len(img_path)):
        temp_img = Image.open(img_path[i])
        resize_image = temp_img.resize((200, 175), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(resize_image)
        # turn the image into a button and put it on the canvas

        button = tk.Button(
            canvas, image=tk_img, command=lambda img_path=img_path[i]: button_click(img_path))
        button.image = tk_img

        canvas.create_window(0, i * 175, anchor='nw',
                             window=button, width=200, height=175)

        print(img_path[i])

    canvas.pack()
    canvas.config(yscrollcommand=scrollbar.set)

    temp_window.mainloop()


def button_click(image_path):
    global BACK_FLAG
    BACK_FLAG = 1
    
    temp_img_index = tk.Label(temp_window, text="Image index: {}".format(temp_index[img_path.index(image_path)]))
    temp_img_index.place(x=WINDOW_SIZE/2-60, y=WINDOW_SIZE/2+115)

    print(f"Button clicked for image: {image_path}")


dirpath = r"./LabelTool/backup27"
result = [os.path.join(dirpath, f) for f in os.listdir(
    dirpath) if os.path.isfile(os.path.join(dirpath, f))]

path1 = "./LabelTool/train"
path2 = "./LabelTool/val"
path3 = "./LabelTool/train/images"
path4 = "./LabelTool/train/labels"
path5 = "./LabelTool/val/images"
path6 = "./LabelTool/val/labels"

if (not (os.path.exists(path1))):
    os.mkdir(path1)
if (not (os.path.exists(path2))):
    os.mkdir(path2)
if (not (os.path.exists(path3))):
    os.mkdir(path3)
if (not (os.path.exists(path4))):
    os.mkdir(path4)
if (not (os.path.exists(path5))):
    os.mkdir(path5)
if (not (os.path.exists(path6))):
    os.mkdir(path6)


window = tk.Tk()
window.title('Score')
window.geometry("{}x{}".format(WINDOW_SIZE, WINDOW_SIZE))

button = tk.Button(window, text="Send", command=send_score)
user_input = tk.Entry(window, width=USER_INPUT_BAR_SIZE)
input_title = tk.Label(window, text="拉花評分: ", font=("Arial", 10))
input_remind = tk.Label(
    window, text="(請輸入介於0~10分的整數，-1表示無效資料)", font=("Arial", 10))
back_button = tk.Button(window, text="Back", command=back_menu)


score_judge()

window.bind('<Return>', send_score)
# temp_window.bind('Return>', write_score)
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
