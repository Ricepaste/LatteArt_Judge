import socket
from tkinter import filedialog
import tkinter as tk
from PIL import Image, ImageTk

WINDOW_HEIGHT = 500
WINDOW_WIDTH = 500
LABEL_HEIGHT = 30
LABEL_WIDTH = 300
ENTRY_HEIGHT = 30
ENTRY_WIDTH = 150
ENTRY_BUTTON_HEIGHT = 30
ENTRY_BUTTON_WIDTH = 100
BROWSE_BUTTON_HEIGHT = 30
BROWSE_BUTTON_WIDTH = 175
IMAGE_P_HEIGHT = 300
IMAGE_P_WIDTH = 300
SENT_BUTTON_HEIGHT = 30
SENT_BUTTON_WIDTH = 100


def browse_clicked():
    # label.config(text="Hello World!")
    global filename, image_p, tk_image
    filename = filedialog.askopenfilename()
    image = Image.open(filename).convert('RGB')
    long_side = max(image.size)
    ratio = 300/long_side
    image = image.resize(
        (int(image.size[0]*ratio), int(image.size[1]*ratio)), Image.LANCZOS)
    tk_image = ImageTk.PhotoImage(image)

    image_p = tk.Label(window, image=tk_image)
    image_p.place(x=10, y=LABEL_HEIGHT + ENTRY_HEIGHT +
                  ENTRY_BUTTON_HEIGHT+BROWSE_BUTTON_HEIGHT+20, height=300, width=300)
    window.update()


def entry_button_clicked():
    global IP, client
    IP = entry.get()
    print(IP)

    # 建立一個socket
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 主動去連線區域網內IP為192.168.27.238，埠為6688的程序
    client.connect((IP, 8000))
    label.config(text="TCP connected")
    entry_button.config(state='disabled', text="Success")


def sent_clicked():
    global filename, client
    # im = open(".\\main\\TCP_photo\\test.jpg", mode='rb')
    im = open(filename, mode='rb')
    im_byte = im.read()
    # 傳送資料
    # client.sendall(data)
    client.sendall(im_byte)
    # 接收服務端的反饋資料
    rec_data = client.recv(1024)
    print(b'form server receive:' + rec_data)
    im.close()


# global image_p
window = tk.Tk()
window.title('Latte Art Score Judge')
window.minsize(width=500, height=500)
window.resizable(width=False, height=False)

label = tk.Label(text="TCP connecting", font=("Arial", 14, "bold"),
                 padx=5, pady=5, fg="black")
entry = tk.Entry(width=30, font=("Arial", 14, "bold"),
                 fg="black", state='normal')
entry.insert(tk.END, string="192.168.137.1")
entry_button = tk.Button(text="confirmed", font=("Arial", 14, "bold"), padx=5,
                         pady=5, bg="gray", fg="black", command=entry_button_clicked)
browse_button = tk.Button(text="Browse Picture", font=("Arial", 14, "bold"), padx=5,
                          pady=5, bg="blue", fg="light green", command=browse_clicked)
sent_button = tk.Button(text="Sent", font=("Arial", 14, "bold"), padx=5,
                        pady=5, bg="blue", fg="light green", command=sent_clicked)
image_p = tk.Label()

# label.pack(side=tk.LEFT)
# entry.pack(side=tk.LEFT)
# entry_button.pack(side=tk.TOP)
# browse_button.pack(side=tk.TOP)
label.place(x=WINDOW_WIDTH/2-LABEL_WIDTH/2, y=0,
            height=LABEL_HEIGHT, width=LABEL_WIDTH)
entry.place(x=10, y=LABEL_HEIGHT,
            height=ENTRY_HEIGHT, width=ENTRY_WIDTH)
entry_button.place(x=ENTRY_WIDTH+20, y=LABEL_HEIGHT,
                   height=ENTRY_BUTTON_HEIGHT, width=ENTRY_BUTTON_WIDTH)
browse_button.place(x=10, y=LABEL_HEIGHT +
                    ENTRY_HEIGHT+10, height=BROWSE_BUTTON_HEIGHT, width=BROWSE_BUTTON_WIDTH)
image_p.place(x=10, y=LABEL_HEIGHT + ENTRY_HEIGHT +
              ENTRY_BUTTON_HEIGHT+BROWSE_BUTTON_HEIGHT+20, height=300, width=300)
sent_button.place(x=10, y=LABEL_HEIGHT + ENTRY_HEIGHT + ENTRY_BUTTON_HEIGHT +
                  BROWSE_BUTTON_HEIGHT+IMAGE_P_HEIGHT+30, height=SENT_BUTTON_HEIGHT, width=SENT_BUTTON_WIDTH)
print(entry.get())
window.mainloop()

# 傳送資料告訴伺服器退出連線
client.close()


# # global filename, IP
# # im = open(".\\main\\TCP_photo\\test.jpg", mode='rb')
# im = open(filename, mode='rb')
# im_byte = im.read()


# # 建立一個socket
# client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# # 主動去連線區域網內IP為192.168.27.238，埠為6688的程序
# client.connect((IP, 8000))

# while True:
#     # 接受控制檯的輸入
#     data = input()
#     # 對資料進行編碼格式轉換，不然報錯
#     data = data.encode('utf-8')
#     # 如果輸入quit則退出連線
#     if data == b'quit':
#         print(b'connect quit.')
#         break
#     else:
#         # 傳送資料
#         # client.sendall(data)
#         client.sendall(im_byte)
#         # 接收服務端的反饋資料
#         rec_data = client.recv(1024)
#         print(b'form server receive:' + rec_data)

# # 傳送資料告訴伺服器退出連線
# client.sendall(b'quit')
# client.close()

# _ = input('Press Enter to continue...')
