import socket
from module.camera import Camera
import numpy as np
import pickle
from module.cam_server import CamServer
from utils.file import save_file
# Define task constants
ACTION_DONE = 10
END = 0
START = 1
ACTION_DONE = 10
IN_PROGRESS = 20
MOVE_ARM_FOR_CALI_TASK = 3
MOVE_ARM_RANDOM_TASK = 4
MOVE_ARM_FOR_GRAP_TASK = 5
COLLECT_JOINT1_FOR_NN = 6
PICK_LEAF = 7
CONTINUE = 2



# Directory where files are saved
directories = {
    'rgb': 'rgb_cali/',
    'depth': 'depth_cali',
    'eye_to_hand': 'eye_to_hand/',
    'joint1_nn': 'joint1_nn/'
}

    

progress = START
ip_port = ('192.168.101.11', 4000)

# Create socket

sk = socket.socket()

sk.connect(ip_port)
sk.settimeout(500)

data = sk.recv(1024).decode()
kinect = Camera()
progress = START
block_center_list = []

# Main loop
while True: 
    # Receive welcome message from server
    if progress == START:
        task = input("Enter task number: ")
        sk.sendall(str(task).encode())
    data = sk.recv(1024).decode()
    try:
        task = int(task)
    except ValueError:
        print("Invalid task number. Please enter a valid task number.")
        continue
    
    if task == MOVE_ARM_FOR_CALI_TASK:
        progress = IN_PROGRESS
        # Add logic for handling MOVE_ARM_FOR_CALI_TASK
        print("Performing MOVE_ARM_FOR_CALI_TASK...")
        if data == 'capture':
            print('Capturing image...')
            rgb_img = kinect.capture_rgb_img()
            save_file(directories,rgb_img,'rgb')

        if data == str(ACTION_DONE):
            print("Task is finished")
            progress = START
            continue
    if task == COLLECT_JOINT1_FOR_NN:
        progress = IN_PROGRESS
        # Add logic for handling MOVE_ARM_FOR_CALI_TASK
        print("Performing COLLECT_JOINT1_FOR_NN...")
        block_centers = []
        if data == 'capture':
            for i in range(5):    
                rgb_img = kinect.capture_rgb_img()
                depth_img = kinect.capture_depth_img()
                block_center = kinect.execute_task(rgb_img, depth_img, 'block')
                block_centers.append(block_center)
            block_center_list.append(np.round(np.median(block_centers,axis=0)))  
            print(block_center_list)
            sk.sendall(str(CONTINUE).encode())
        if data == str(ACTION_DONE):
            print("Task is finished")
            progress = START
            save_file(directories, block_center_list,'joint1_nn')
            continue
    if task == PICK_LEAF:
    # Add handling for other tasks (MOVE_ARM_RANDOM_TASK, MOVE_ARM_FOR_GRAP_TASK, etc.)
        progress = IN_PROGRESS
        print("Performing PICK_LEAF...")
        leaf_center = []
        leaf_centers = []
        if data == 'capture':
            for i in range(5):    
                print(i)
                rgb_img = kinect.capture_rgb_img()
                depth_img = kinect.capture_depth_img()
                leaf_center = kinect.execute_task(rgb_img, depth_img, 'block')
                leaf_centers.append(leaf_center)
            leaf_center = np.round(np.median(leaf_centers,axis=0)).tolist()
                # Serialize the list to JSON format
            data =pickle.dumps(leaf_center)
            # Send the JSON data over the socket connection
            sk.sendall(data)
        if data == str(ACTION_DONE):
            print("Task is finished")
            progress = START
            continue
    if task == END:
        print("Thank you for using, goodbye!")
        break



# Close the socket
sk.close()
