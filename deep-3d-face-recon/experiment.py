from options.test_options import TestOptions
from test_face_recon import test

import cv2
import mtcnn
import dlib
import os
import numpy as np

img_folder = "test_images"
img_name = "pie2"
img_extension = "png"
landmarks_folder = img_folder + "/detections"

# CODE FROM
# https://github.com/JustinGuese/mtcnn-face-extraction-eyes-mouth-nose-and-speeding-it-up/blob/master/MTCNN%20example.ipynb
image = cv2.imread(img_folder + "/" + img_name + "." + img_extension)
detector = mtcnn.MTCNN()
# detect faces in the image
faces = detector.detect_faces(image)

print('FOUND: ' + str(len(faces)) + ' faces')

for face in faces:
    x, y, width, height = face['box']
    cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 3)
    required_landmarks = [
        face['keypoints']['left_eye'],
        face['keypoints']['right_eye'],
        face['keypoints']['nose'],
        face['keypoints']['mouth_left'],
        face['keypoints']['mouth_right']
    ]

    for rl in required_landmarks:
        cv2.circle(image, rl, 4, (0, 0, 255), -1)

    #save landmarks to file
    lm_file = landmarks_folder + "/" + img_name + ".txt"

    if not os.path.exists(landmarks_folder):
        os.makedirs(landmarks_folder)

    with open(lm_file, 'w') as file:
        for lm in required_landmarks:
            file.write(str(lm[0]) + " " + str(lm[1]) + '\n')


# opt = TestOptions().parse()
# opt.img_folder = img_folder
# opt.epoch = 20
# opt.name = "pretrained"
# test(0, opt,opt.img_folder)
#
cv2.imshow("Image", image)
cv2.waitKey(0)