import socket
import os


# current path
path = os.getcwd()
print(path)

# 建立一個socket套接字，該套接字還沒有建立連線
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
IP = socket.gethostbyname(socket.gethostname())
# IP = '10.115.48.186'
print(IP)
server.bind((IP, 8000))

# 開始監聽，並設定最大連線數
server.listen(5)

print(u'waiting for connect...')
# 等待連線，一旦有客戶端連線後，返回一個建立了連線後的套接字和連線的客戶端的IP和埠元組
connect, (host, port) = server.accept()
print(u'the client %s:%s has connected.' % (host, port))


while True:
    # 接受客戶端的資料
    ALL_Data = b''
    try:
        data = connect.recv(262144)
    except:
        print("error")
        server.close()
        break
    ALL_Data += data
    print(len(data))
    while (len(data) == 262144):
        data = connect.recv(262144)
        ALL_Data += data
        print(len(data))
    # data = connect.recv(262144)
    # ALL_Data += data
    # print(len(data))
    print("all data size:\t{}bytes".format(len(ALL_Data)))

    if data == b'quit' or data == b'':
        print(b'the client has quit.')
        break
    else:
        # 傳送資料給客戶端
        connect.sendall(b'your words has received.')
        # print(b'the client say:' + data)

    img = open(
        "C:\\Users\\sharpaste\\Documents\\program\\testing\\Python\\yolo\\main\\TCP_photo\\test.jpg", mode='wb')
    if (data != b'' and data != b'quit'):
        img.write(ALL_Data)
    img.close()

    # data = data.decode('utf-8')
    # 如果接受到客戶端要quit就結束迴圈

# 結束socket
server.close()
# _ = input('Press Enter to continue...')
