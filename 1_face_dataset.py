import cv2
import os
import time

# Directory where images will be stored
directory = fr"/media/kumar/HDD1/INFIDATA/EXPERIMENT LAB/face-recognition-module/face_dataset/"

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Initialize webcam
cam = cv2.VideoCapture(0)

print("/n [INFO] Initializing face capture. Look at the camera and wait...")

username = input("Please enter username: ")

# Create a directory for the user's images
user_directory = os.path.join(directory, username)
if not os.path.exists(user_directory):
    os.makedirs(user_directory)

# Initialize individual sampling face count
time.sleep(3)
count = 0

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)  # flip video image vertically
    count += 1
    cv2.imwrite(os.path.join(user_directory, f"{count}.jpg"), img)
    cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video

    if k == 27:
        break
    elif count >= 50:  # Take 50 face samples and stop video
        break

# Release webcam and close OpenCV windows
print("/n [INFO] Exiting program and cleaning up")
cam.release()
cv2.destroyAllWindows()
