import sys
import os
sys.path.append('../../../deep-3d-face-recon')

from termcolor import cprint

from options.test_options import TestOptions
from test_face_recon import get_coeffs_from_image

from pipeline_util.enums import ComparisonMethods
from pipeline_modules.context import FailedTestingEntry
from pipeline_modules.context import Context
from pipeline.pipeline import NextStep


class Deep3DCoefficientGenerator:
    """Runs the deep 3d face_recon network and saves the coefficients to the context var"""
    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('Deep3DCoefficientGenerator: started', 'cyan')
        opt = TestOptions().parse(print_options=False)
        opt.img_folder = context.working_dir_path
        opt.epoch = 20
        opt.checkpoints_dir = '../deep-3d-face-recon/checkpoints'
        opt.net_recog_path = '../deep-3d-face-recon/checkpoints/recog_model/ms1mv3_arcface_r50_fp16/backbone.pth'
        opt.init_path = '../deep-3d-face-recon/checkpoints/init_model/resnet50-0676ba61.pth'
        opt.bfm_folder = '../deep-3d-face-recon/BFM_2009'
        opt.name = 'pretrained'  # TODO Future: get model opts globally

        try:
            context.deep_3d_coeffs = get_coeffs_from_image(0, opt, opt.img_folder)  # dict id -> file name

            # Check if some images failed and save as failed entry if some did
            all_image_names = os.listdir(context.working_dir_path)
            if len(all_image_names) > len(context.deep_3d_coeffs):
                for file_name in all_image_names:
                    if not '.' in file_name:
                        continue  # skip directories
                    if file_name.split('.')[0] not in context.deep_3d_coeffs:
                        cprint('Failed to reconstruct 3D Shape of ' + file_name, 'red')
                        context.failed_testing_entries.append(FailedTestingEntry(
                            ComparisonMethods.COEFFICIENT_BASED_3D.name,
                            -1,
                            'reconstruction'
                        ))

            cprint('Deep3DCoefficientGenerator: done', 'green')
            next_step(context)
        except Exception as error:
            cprint('Failed getting 3D Face coefficients', 'red')
            raise ValueError(error) from error