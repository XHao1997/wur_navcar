import socket
import time
import struct

HOST = '192.168.1.103'
PORT = 8888

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))

while True:
    data, address = sock.recvfrom(1024)
    print(struct.unpack('i', data))

sock.close()