from typing import List
import shutil
import os
import cv2
import dlib

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep


class DatPreparation3D:
    """Collects all Images from Input Directory, moves them to
    working dir and detects landmarks for later deep 3D Face Recon use"""
    def __init__(self, detection_subdir_name: str, dat_file_path: str) -> None:
        self._detection_subdir_name = detection_subdir_name
        self._dat_file_path = dat_file_path
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(dat_file_path)


    def __call__(self, context: Context, next_step: NextStep) -> None:
        print("------------------------------------")
        print("DataPreparation3D: started")

        #clean working dir
        shutil.rmtree(context.working_dir_path + '/')

        # Create Detections dir if not exist
        complete_detection_dir_path = context.working_dir_path + "/" + self._detection_subdir_name
        if not os.path.exists(complete_detection_dir_path):
            os.makedirs(complete_detection_dir_path)

        # TODO limit size of images
        # Get All Images, move them to working dir and create the respective detection landmark file
        image_file_names = os.listdir(context.input_dir_path)
        print("DataPreparation3D: found " + str(len(image_file_names)) + " images in input dir")
        for file_name in image_file_names:
            # Move file to working dir
            shutil.copy(os.path.join(context.input_dir_path, file_name), context.working_dir_path)
            image = cv2.imread(context.working_dir_path + "/" + file_name)
            face_rects = self._detector(image)
            face_rect = face_rects[0] # TODO print a warning if more than 1 face on img
            required_landmarks = self.get_required_landmarks(image, face_rect)
            image_name = file_name.split('.')[0]
            detection_file_path = complete_detection_dir_path + "/" + image_name + ".txt"
            self.write_landmarks_to_file(detection_file_path, required_landmarks)

        print("DataPreparation3D: done")
        next_step(context)


    def get_required_landmarks(self, image, face_rect):
        x1 = face_rect.left()
        y1 = face_rect.top()
        x2 = face_rect.right()
        y2 = face_rect.bottom()
        shape = self._predictor(image, face_rect)
        # Refered to dlib documentation to get the correct landmark indexes
        return [
            shape.part(39),
            shape.part(42),
            shape.part(30),
            shape.part(48),
            shape.part(54)
        ]


    def write_landmarks_to_file(self, detection_file_path, landmarks):
        with open(detection_file_path, 'w') as file:
            for lm in landmarks:
                file.write(str(lm.x) + " " + str(lm.y) + '\n')