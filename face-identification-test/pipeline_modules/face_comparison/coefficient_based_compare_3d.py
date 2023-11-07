from termcolor import cprint
import numpy as np
from numpy.linalg import norm

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep
from pipeline_modules.context import TestingResultEntry
from pipeline_util.enums import ComparisonMethods

class CoefficientBasedCompare3D:
    """Compares two 3D deep face_recon Coefficient Vectors with cosine similarity"""
    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('CoefficientBasedCompare3D: started', 'cyan')

        # Loop all open testing entries, get 3d coeffs and save comparison result to testing results
        print('comparing ' + str(len(context.open_testing_entry.items())) + ' image pairs')
        for id, testing_entry in context.open_testing_entry.items():
            gallery_image_coeffs = context.deep_3d_coeffs[
                testing_entry.gallery_image_file_name.split('.')[0]
            ]
            input_image_coeffs = context.deep_3d_coeffs[
                testing_entry.input_image_file_name.split('.')[0]
            ]
            gallery_image_vector = np.concatenate([
                gallery_image_coeffs['id'].cpu().numpy()[0],
                gallery_image_coeffs['tex'].cpu().numpy()[0]
            ])
            input_image_vector = np.concatenate([
                input_image_coeffs['id'].cpu().numpy()[0],
                input_image_coeffs['tex'].cpu().numpy()[0]
            ])

            cosine_similarity = self.cosine_similarity(gallery_image_vector, input_image_vector)
            context.testing_result_entries.append(TestingResultEntry(
                open_testing_entry_id=id,
                method=ComparisonMethods.COEFFICIENT_BASED_3D,
                prediction=cosine_similarity
            ))

        cprint('CoefficientBasedCompare3D: done', 'green')
        next_step(context)

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (norm(a) * norm(b))
