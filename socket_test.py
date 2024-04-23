import socket
from module.camera import Camera
import os
import cv2
# Define task constants
ACTION_DONE = 10
END = 0
START = 1
IN_PROGRESS = 2
MOVE_ARM_FOR_CALI_TASK = 3
MOVE_ARM_RANDOM_TASK = 4
MOVE_ARM_FOR_GRAP_TASK = 5
# Directory where files are saved
directory_rgb = 'rgb_cali/'
directory_depth = 'depth_cali'
directory_eye_to_hand = 'eye_to_hand/'
# Get a list of files in the directory
files = os.listdir(directory_eye_to_hand)
f = 0
# Filter only files with specific extensions, like jpg, png, etc.
image_files = [file for file in files if file.endswith('.jpg') or file.endswith('.png')]

# Find the maximum number in the filenames
max_num = max([int(file.split('_')[1].split('.')[0]) for file in image_files]) if image_files else 0

progress = START
ip_port = ('192.168.101.11', 5000)

# Create socket

sk = socket.socket()

sk.connect(ip_port)
sk.settimeout(500)

data = sk.recv(1024).decode()
# print('Server:', data)
kinect = Camera()
progress = START
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
        # Receive response from server
        # data = sk.recv(1024).decode()
        # print('Server:', data)

        if data == 'capture':
            new_filename_rgb = f"rgb_{max_num + f:02d}.png"
            new_filename_depth = f"depth_{max_num + f:02d}.png"
            print('Capturing image...')
            rgb_img = kinect.get_video()
            f+=1
            cv2.imwrite(os.path.join(directory_eye_to_hand, new_filename_rgb),rgb_img)
            # Add logic for capturing image
            # kinect.save_img(current_img)
            

        if data == str(ACTION_DONE):
            print("Task is finished")
            progress = START
            continue

    # Add handling for other tasks (MOVE_ARM_RANDOM_TASK, MOVE_ARM_FOR_GRAP_TASK, etc.)

    if task == END:
        print("Thank you for using, goodbye!")
        break



# Close the socket
sk.close()
