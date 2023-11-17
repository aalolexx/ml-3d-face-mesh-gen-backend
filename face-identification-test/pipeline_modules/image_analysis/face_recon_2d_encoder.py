from termcolor import cprint
import face_recognition

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep


class FaceRecon2DEncoder:
    """Runs the 2D face_recon network and saves the encodings to context var"""
    def __init__(self, include_vpn_images: bool, vpn_images_subdir: str) -> None:
        self._include_vpn_images = include_vpn_images
        self._vpn_images_subdir = vpn_images_subdir

    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('FaceRecon2DEncoder: started', 'cyan')

        failed_images = []

        for id, testing_entry in context.open_testing_entry.items():
            try:
                gallery_image_path = context.working_dir_path + '/' + testing_entry.gallery_image_file_name
                input_image_path = context.working_dir_path + '/' + testing_entry.input_image_file_name
                gallery_image = face_recognition.load_image_file(gallery_image_path)
                input_image = face_recognition.load_image_file(input_image_path)
            except Exception as error:
                cprint('Could not find all required images for: ' + str(id), 'red')
                continue

            # TODO check how to do this on GPU
            gallery_image_encodings = face_recognition.face_encodings(gallery_image)
            input_image_encodings = face_recognition.face_encodings(input_image)
            gallery_image_name = testing_entry.gallery_image_file_name.split('.')[0]
            input_image_name = testing_entry.input_image_file_name.split('.')[0]

            self.add_encoding_to_context(context, gallery_image_encodings, gallery_image_name)
            self.add_encoding_to_context(context, input_image_encodings, input_image_name)

        # Now for the VPN Image, if required
        if self._include_vpn_images:
            for id, testing_entry in context.open_testing_entry.items():
                try:
                    vpn_path_prefix = context.working_dir_path + '/' + self._vpn_images_subdir + '/vpn_'
                    gallery_vpn_image_path = vpn_path_prefix + testing_entry.gallery_image_file_name
                    input_vpn_image_path = vpn_path_prefix + testing_entry.input_image_file_name
                    gallery_vpn_image = face_recognition.load_image_file(gallery_vpn_image_path)
                    input_vpn_image = face_recognition.load_image_file(input_vpn_image_path)
                except Exception as error:
                    cprint('Could not find all required VPN images for: ' + str(id), 'red')
                    continue

                gallery_vpn_image_encodings = face_recognition.face_encodings(gallery_vpn_image)
                input_vpn_image_encodings = face_recognition.face_encodings(input_vpn_image)
                gallery_vpn_image_name = 'vpn_' + testing_entry.gallery_image_file_name.split('.')[0]
                input_vpn_image_name = 'vpn_' + testing_entry.input_image_file_name.split('.')[0]
                self.add_encoding_to_context(context, gallery_vpn_image_encodings, gallery_vpn_image_name)
                self.add_encoding_to_context(context, input_vpn_image_encodings, input_vpn_image_name)

        # TODO protocol failed entries

        cprint('FaceRecon2DEncoder: done', 'green')
        next_step(context)


    def add_encoding_to_context(self, context, image_encoding, image_name):
        if len(image_encoding) > 1:
            cprint('found more than 1 face in image at ' + image_name + '. Parsing face 0', 'yellow')

        if len(image_encoding) < 1:
            cprint('Could not locate a face in image ' + image_name, 'red')
            context.face_recognition_2d_encodings[image_name] = None
        else:
            context.face_recognition_2d_encodings[image_name] = image_encoding[0]