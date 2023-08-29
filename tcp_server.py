from collections import deque
import socket
import json
import threading
from time import sleep

import select
import queue


class TCPClient(object):
    def __init__(self):
        self.tcpIp = '127.0.0.1'
        self.tcpPort = 5555
        self.buffer_size = 99999

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.inputCheck = []
        self.outputCheck = []
        self.isChecking = False
        self.isAcquiring = False
        self.msgQueue = queue.Queue()
        self.myDeque = deque([0]*10000, maxlen=10000)  # length of queue, record 10s datas (fs = 1000Hz)
        # when 'maxlen' is decided, deque can put and pop items automatically

    def connect(self):
        self.socket.connect((self.tcpIp, self.tcpPort))
        self.outputCheck.append(self.socket)
        self.isChecking = True

    def start(self, d):
        thread = threading.Thread(target=self.msg_checker, args=(d,))
        thread.daemon = True
        thread.start()

    def stop(self):
        self.isChecking = False
        self.socket.close()

    def msg_checker(self, d):
        while self.isChecking:
            readable, writable, exceptional = select.select(self.inputCheck, self.outputCheck, self.inputCheck)
            for s in readable:
                message = s.recv(self.buffer_size)
                if not self.isAcquiring:
                    self.inputCheck = []
                else:
                    message = json.loads(message)
                    message = message["returnData"]
                    # for device in message.keys():
                    #     for data_list in message[device]:
                    #         number = data_list[-1]  # this is the data we collected
                    #         self.myDeque.append(number)
                    #         d[:] = self.myDeque  # add datas to the shared queue
                    #         # print(number)  # Print the number

                    try:
                        for device in message.keys():
                            for data_list in message[device]:
                                number = data_list[-1]  # this is the data we collected
                                self.myDeque.append(number)
                                d[:] = self.myDeque  # add datas to the shared queue
                                # print(number)  # Print the number
                    except AttributeError as e:
                        print(f"Error: {e}. Message content: {message}")

            for _ in writable:
                try:
                    next_msg = self.msgQueue.get_nowait()
                except queue.Empty:
                    pass
                else:
                    self.socket.send(str(next_msg).encode())

            for s in exceptional:
                print("exceptional ", s)

    def add_msg_to_send(self, data):
        self.msgQueue.put(data)
        if self.socket not in self.outputCheck:
            self.outputCheck.append(self.socket)
        if self.socket not in self.inputCheck:
            self.inputCheck.append(self.socket)

    def set_is_acquiring(self, is_acquiring):
        self.isAcquiring = is_acquiring


def tcp_client_processing(d, q):
    CONNECTION = TCPClient()
    connect_flag = False
    while not connect_flag:
        try:
            CONNECTION.connect()
            connect_flag = True
        except ConnectionRefusedError:
            sleep(3)

    CONNECTION.start(d)
    while True:

        user_action = str(q.get())  # get input from dash app
        if user_action == '0':
            CONNECTION.set_is_acquiring(True)
        elif user_action == '1':
            CONNECTION.set_is_acquiring(False)
        elif user_action == '2':
            CONNECTION.stop()
            break
        new_msg = action_decode(user_action)
        CONNECTION.add_msg_to_send(new_msg)  # send actions to opensignals app


def action_decode(action):
    if action == '0':
        return 'start'
    elif action == '1':
        return 'stop'
    elif action == '2':
        return ''
    else:
        return ''
