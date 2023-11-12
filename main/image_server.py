import socket
import os
import select
import threading


def handle_client(client_socket, address):
    print(f'Accepted connection from {address}')

    while True:
        ALL_Data = b''
        try:
            data = client_socket.recv(1024)
        except:
            print("Error receiving data.")
            client_socket.close()
            break

        ALL_Data += data
        print(len(data))
        while len(data) == 1024:
            data = client_socket.recv(1024)
            ALL_Data += data
            print(len(data))

        print(f"All data size: {len(ALL_Data)} bytes")

        if data == b'quit' or data == b'':
            print(b'The client has quit.')
            client_socket.close()
            break
        else:
            client_socket.sendall(b'Your words have been received.')

        img_path = os.path.join(os.getcwd(), "received_image.jpg")
        img = open(img_path, mode='wb')
        if data != b'' and data != b'quit':
            img.write(ALL_Data)
        img.close()

    print(f'Connection from {address} closed.')


def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    IP = socket.gethostbyname_ex(socket.gethostname())
    print(IP[2][2])
    server.bind((IP[2][2], 8000))
    server.listen(5)

    print('Waiting for connections...')

    while True:
        try:
            readable, _, _ = select.select([server], [], [], 0.1)

            for s in readable:
                if s is server:
                    client_socket, client_address = server.accept()

                    client_handler = threading.Thread(
                        target=handle_client, args=(client_socket, client_address))
                    client_handler.start()

        except KeyboardInterrupt:
            print('Server shutting down...')
            break

    server.close()


if __name__ == "__main__":
    main()
