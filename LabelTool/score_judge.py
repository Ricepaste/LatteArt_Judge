import os
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox

IMAGE_SIZE = 300
WINDOW_SIZE = 500
USER_INPUT_BAR_SIZE = 10


def import_image(path):
    global img
    img = Image.open(path)
    resize_image = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
    tk_img = ImageTk.PhotoImage(resize_image)
    lab = tk.Label(window, image=tk_img)
    lab.image = tk_img  # 保留對圖像物件的引用以避免被垃圾回收
    return lab


i = 0


def score_judge():
    global curr_img
    curr_img = import_image(result[i])
    curr_img.pack()

    button.place(x=WINDOW_SIZE/2+USER_INPUT_BAR_SIZE*7/2+15, y=IMAGE_SIZE+45)

    user_input.place(x=WINDOW_SIZE/2-USER_INPUT_BAR_SIZE*7/2, y=IMAGE_SIZE+50)

    input_title.place(x=WINDOW_SIZE/2-USER_INPUT_BAR_SIZE *
                      7/2-70, y=IMAGE_SIZE+47)

    input_remind.place(x=WINDOW_SIZE/2-USER_INPUT_BAR_SIZE *
                       7/2-20, y=IMAGE_SIZE+80)


def window_close():
    window.destroy()


def send_score():
    # send score to excel to the corresponding index
    global i
    i += 1
    if i == len(result):
        window_close()
    else:
        curr_img.destroy()
        get_num_from_bar()
        if (not (score.isdigit()) or (int(score) < 0 or int(score) > 10)):
            messagebox.showerror(title="錯誤輸入", message="媽的文盲")
            i -= 1
        clearBar()
        score_judge()


def get_num_from_bar():
    global score
    score = user_input.get()
    # print(score)


def clearBar():
    user_input.delete(0, 'end')


dirpath = r".\\LabelTool\\backup27"
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
