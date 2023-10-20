from termcolor import cprint
import face_recognition

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep

# TODO optional: Try to pass detections as var and not as txt file

class FaceRecon2DEncoder:
    """Runs the 2D face_recon network and saves the encodings to context var"""
    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('FaceRecon2DEncoder: started', 'cyan')

        for id, testing_entry in context.open_testing_entry.items():
            gallery_image_path = context.working_dir_path + '/' + str(id) + '_g.jpg'
            input_image_path = context.working_dir_path + '/' + str(id) + '_i.jpg'
            gallery_image = face_recognition.load_image_file(gallery_image_path)
            input_image = face_recognition.load_image_file(input_image_path)
            gallery_image_encodings = face_recognition.face_encodings(gallery_image)
            input_image_encodings = face_recognition.face_encodings(input_image)

            if len(gallery_image_encodings) > 1 or len(input_image_encodings) > 1:
                cprint('found more than 1 face in image with id ' + str(id) + '. Parsing face 0', 'yellow')

            if len(gallery_image_encodings) < 1 or len(input_image_encodings) < 1:
                raise ValueError('Could not locate a face in image with id ' + str(id))

            context.face_recon_2d_encodings[str(id) + '_g'] = gallery_image_encodings[0]
            context.face_recon_2d_encodings[str(id) + '_i'] = input_image_encodings[0]

        cprint('FaceRecon2DEncoder: done', 'green')
        next_step(context)