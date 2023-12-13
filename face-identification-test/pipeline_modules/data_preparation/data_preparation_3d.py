import os
from PIL import Image
import numpy as np
import mtcnn
import matplotlib.pyplot as plt

from termcolor import cprint

from pipeline_modules.context import Context, FailedTestingEntry
from pipeline.pipeline import NextStep
from pipeline_util.enums import ComparisonMethods


class DataPreparation3D:
    """Collects all Images from Working Directory,
    and detects + saves landmarks for later deep 3D Face Recon use"""
    def __init__(self, detection_subdir_name: str) -> None:
        self._detection_subdir_name = detection_subdir_name
        self._detector = mtcnn.MTCNN()


    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('DataPreparation3D: started', 'cyan')

        # Create Detections dir if not exist
        complete_detection_dir_path = context.working_dir_path + '/' + self._detection_subdir_name
        if not os.path.exists(complete_detection_dir_path):
            os.makedirs(complete_detection_dir_path)

        # TODO Future: limit size of images
        # Get All Imagesfrom working dir and create the respective detection landmark file
        image_file_names = os.listdir(context.working_dir_path)
        print('DataPreparation3D: found ' + str(len(image_file_names)) + ' images in working dir')
        for file_name in image_file_names:
            if not '.' in file_name:
                # skip directories
                continue

            full_image_path = context.working_dir_path + '/' + file_name
            image = Image.open(full_image_path)
            image = image.convert("RGB")
            image_np = np.array(image)
            face_rects = self._detector.detect_faces(image_np)

            # The Face may be covered too much or the image is corrupt. in this case warn the user and delete the image
            if len(face_rects) <= 0:
                self.protocol_failed_entry(context, file_name)
            else:
                if len(face_rects) > 1:
                    cprint('Multiple Faces in ' + file_name + ', taking first one', 'yellow')
                face_rect = face_rects[0]
                required_landmarks = self.get_required_landmarks(face_rect)
                image_name = file_name.split('.')[0]
                detection_file_path = complete_detection_dir_path + '/' + image_name + '.txt'
                self.write_landmarks_to_file(detection_file_path, required_landmarks)

        cprint('DataPreparation3D: done', 'green')
        next_step(context)


    def get_required_landmarks(self, face_rect):
        # Refered to dlib documentation to get the correct landmark indexes
        return [
            face_rect['keypoints']['left_eye'],
            face_rect['keypoints']['right_eye'],
            face_rect['keypoints']['nose'],
            face_rect['keypoints']['mouth_left'],
            face_rect['keypoints']['mouth_right']
        ]


    def write_landmarks_to_file(self, detection_file_path, landmarks):
        with open(detection_file_path, 'w') as file:
            for lm in landmarks:
                file.write(str(lm[0]) + ' ' + str(lm[1]) + '\n')


    def protocol_failed_entry(self, context, file_name):
        cprint('failed to find a face rect for image: ' + file_name, 'red')
        # Save Failed Testing Entries for all 3D Method since all of them depend on this landmark detection here
        context.failed_testing_entries.append(FailedTestingEntry(
            ComparisonMethods.COEFFICIENT_BASED_3D.name,
            -1,
            'detection'
        ))
