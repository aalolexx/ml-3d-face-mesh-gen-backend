import sys
sys.path.append('../../../deep-3d-face-recon')

from termcolor import cprint
import cv2
import mtcnn
import os
import numpy as np
import matplotlib.pyplot as plt
from pipeline_util.data_util import get_coeff_array_from_coeff_dict

from model_generator import get_model_image

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep


class VPNImageCreator:
    """Creates a Viewpoint Normalize Image (Frontal Face Image) Based of a 3DMM and a stadnard person image"""
    def __init__(self, standard_person_subdir: str) -> None:
        self._standard_person_subdir = standard_person_subdir
        self._detector = mtcnn.MTCNN()

    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('VPNImageCreator: started', 'cyan')

        # TODO Create own average person image
        standard_person_image, standard_face_mtcnn = self.prepare_standard_face(context)

        for id, testing_entry in context.open_testing_entry.items():
            try:
                vpn_input_image_full_path = os.path.abspath(
                    context.working_dir_path + '/vpn_' + testing_entry.input_image_file_name)
                vpn_gallery_image_full_path = os.path.abspath(
                    context.working_dir_path + '/vpn_' + testing_entry.gallery_image_file_name)

                if not os.path.isfile(vpn_input_image_full_path):
                    input_image_3d_coeffs = context.deep_3d_coeffs[testing_entry.input_image_file_name.split('.')[0]]
                    vpn_input_image = self.get_vpn_image_from_3d_coeffs(input_image_3d_coeffs, standard_person_image, standard_face_mtcnn)
                    cv2.imwrite(vpn_input_image_full_path, vpn_input_image)

                if not os.path.isfile(vpn_gallery_image_full_path):
                    gallery_image_3d_coeffs = context.deep_3d_coeffs[testing_entry.gallery_image_file_name.split('.')[0]]
                    vpn_gallery_image = self.get_vpn_image_from_3d_coeffs(gallery_image_3d_coeffs, standard_person_image, standard_face_mtcnn)
                    cv2.imwrite(vpn_gallery_image_full_path, vpn_gallery_image)

            except Exception as error:
                # TODO this error happens at name splitting. check why there are names without a dot
                cprint('Error creating vpn image for ' + str(id) + ', error: ' + str(error))
                continue

        cprint('VPNImageCreator: done', 'green')
        next_step(context)


    def get_vpn_image_from_3d_coeffs(self, coeffs_3d, standard_person_image, standard_face_mtcnn):
        coeff_arr = get_coeff_array_from_coeff_dict(coeffs_3d)
        model_image = get_model_image(coeff_arr)

        # TODO get eyes programatically
        model_eye_pos = [(80, 75), (140, 75)]

        sf_eye_pos = [
            standard_face_mtcnn['keypoints']['left_eye'],
            standard_face_mtcnn['keypoints']['right_eye']
        ]

        model_eye_distance = (model_eye_pos[1][0] - model_eye_pos[0][0])
        sf_eye_distance = (sf_eye_pos[1][0] - sf_eye_pos[0][0])
        scale_factor = sf_eye_distance / model_eye_distance

        # Fit eye distance of model image to standard image for later overlay
        model_image = cv2.normalize(model_image, None, 0, 255, cv2.NORM_MINMAX)
        model_image = model_image.astype(standard_person_image.dtype)
        model_image = cv2.resize(model_image, (0, 0), fx=scale_factor, fy=scale_factor)

        # Create a mask (The deep3D Model already has a black background)
        model_mask = np.copy(model_image)
        _, model_mask = cv2.threshold(model_mask, 1, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5))
        model_mask = cv2.erode(model_mask, kernel, iterations=3)

        # Do the face swap
        vpn_input_image = cv2.seamlessClone(
            model_image,
            standard_person_image,
            model_mask,
            standard_face_mtcnn['keypoints']['nose'],
            cv2.NORMAL_CLONE
        )

        vpn_input_image = cv2.cvtColor(vpn_input_image, cv2.COLOR_BGR2RGB)
        vpn_input_image = cv2.resize(vpn_input_image, (200, 200))  # resize to save storage
        return vpn_input_image


    def prepare_standard_face(self, context):
        tmp_image = cv2.imread(context.misc_dir_path + '/' + self._standard_person_subdir)
        standard_person_image = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2RGB)
        standard_face_mtcnn = self._detector.detect_faces(standard_person_image)[0]
        return standard_person_image, standard_face_mtcnn