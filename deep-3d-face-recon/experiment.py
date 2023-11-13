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

# datFile = "./dlib/shape_predictor_68_face_landmarks.dat"
#
# # LANDMARK DETECTION CODE SRC
# # https://dontrepeatyourself.org/post/how-to-detect-face-landmarks-with-dlib-python-and-opencv/
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(datFile)
#
# image = cv2.imread(img_folder + "/" + img_name + "." + img_extension)
# # image = cv2.resize(image, (600, 500))
# image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # TODO check if its neccessary to convert to gray
#
# # TODO scale image
#
# # detect the faces
# face_rects = detector(image_gray)
#
# # go through the face bounding boxes
# # TODO Warn when there are multiple faces found
# for rect in face_rects:
#     # extract the coordinates of the bounding box
#     x1 = rect.left()
#     y1 = rect.top()
#     x2 = rect.right()
#     y2 = rect.bottom()
#
#     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
#
#     # apply the shape predictor to the face ROI
#     shape = predictor(image_gray, rect)
#
#     required_landmarks = [
#         shape.part(39),
#         shape.part(42),
#         shape.part(30),
#         shape.part(48),
#         shape.part(54)
#     ]
#
#     for n in range(0, 68):
#         x = shape.part(n).x
#         y = shape.part(n).y
#         cv2.circle(image, (x, y), 4, (255, 0, 0), -1)
#
#     for rl in required_landmarks:
#         cv2.circle(image, (rl.x, rl.y), 4, (0, 0, 255), -1)
#
#     # save landmarks to file
#     lm_file = landmarks_folder + "/" + img_name + ".txt"
#
#     if not os.path.exists(landmarks_folder):
#         os.makedirs(landmarks_folder)
#
#     with open(lm_file, 'w') as file:
#         for lm in required_landmarks:
#             file.write(str(lm.x) + " " + str(lm.y) + '\n')

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