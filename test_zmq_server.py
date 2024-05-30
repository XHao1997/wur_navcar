#
#   Hello World server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#
import copy
import time

import zmq
import threading
from utils import ssh


class Communicator:
    def __init__(self):
        self.context = zmq.Context()
        self.pair_arm = self.context.socket(zmq.REQ)
        self.pair_arm.bind("tcp://192.168.101.14:5556")

    def get_data_from_robot(self):
        return self.pair_arm.recv()


if __name__ == '__main__':
    context = zmq.Context()
    pair_arm = context.socket(zmq.REQ)
    pair_arm.bind("tcp://192.168.101.14:3333")
    ssh.run_remote_stream()
    print('done')
    while True:
        pair_arm.send_pyobj('hi')
        msg = pair_arm.recv_pyobj()
        print(msg)

