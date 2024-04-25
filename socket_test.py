import socket
from module.camera import Camera
import os
import cv2
import numpy as np

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
CONTINUE = 2



# Directory where files are saved
directories = {
    'rgb': 'rgb_cali/',
    'depth': 'depth_cali',
    'eye_to_hand': 'eye_to_hand/',
    'joint1_nn': 'joint1_nn/'
}

def find_file_maxnum(directory):
    files = os.listdir(directory)
    # Find the maximum number in the filenames
    # Filter only files with specific extensions, like jpg, png, npy etc.
    image_files = [file for file in files 
                    if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.npy')]
    max_num = max([int(file.split('_')[1].split('.')[0]) for file in image_files]) if image_files else 0
    return max_num

def set_file_name(directories, key):
    filename = None
    max_num = find_file_maxnum(directories[key])
    if key == 'rgb':
        filename = f"rgb_{max_num + 1:02d}.png"
    elif key == 'depth':
        filename = f"depth_{max_num + 1:02d}.png"
    elif key == 'eye_to_hand':
        filename = f"eye_to_hand_{max_num + 1:02d}"      
    elif key == 'joint1_nn':
        filename = f"joint1_nn_{max_num + 1:02d}"
    return filename   

def save_file(directories, file, key):
    filename = set_file_name(directories, key) 
    if key == 'rgb':
        cv2.imwrite(os.path.join(directories[key], filename),file)
    elif key == 'depth':
        cv2.imwrite(os.path.join(directories[key], filename),file)
    elif key == 'eye_to_hand':
        filename = np.save(os.path.join(directories[key], filename),file)  
    elif key == 'joint1_nn':
        filename = np.save(os.path.join(directories[key], filename),file)  

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
    # Add handling for other tasks (MOVE_ARM_RANDOM_TASK, MOVE_ARM_FOR_GRAP_TASK, etc.)

    if task == END:
        print("Thank you for using, goodbye!")
        break



# Close the socket
sk.close()
