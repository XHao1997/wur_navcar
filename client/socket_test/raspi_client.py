import socket
import struct

HOST = '192.168.1.103'
PORT = 8888

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.connect((HOST, PORT))

while True:
    sock.sendall(struct.pack('i', 1))

sock.close()