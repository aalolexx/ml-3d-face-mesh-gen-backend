import sys
sys.path.append('../../deep-3d-face-recon')

from options.test_options import TestOptions
from test_face_recon import get_coeffs_from_image

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep

class Deep3DCoefficientGenerator:
    """Runs the deep 3d face_recon network and saves the coefficients to a file"""
    def __call__(self, context: Context, next_step: NextStep) -> None:
        print("------------------------------------")
        print("Deep3DCoefficientGenerator: started")
        opt = TestOptions().parse()
        opt.img_folder = context.working_dir_path
        opt.epoch = 20
        opt.checkpoints_dir = '../deep-3d-face-recon/checkpoints'
        opt.net_recog_path = '../deep-3d-face-recon/checkpoints/recog_model/ms1mv3_arcface_r50_fp16/backbone.pth'
        opt.init_path = '../deep-3d-face-recon/checkpoints/init_model/resnet50-0676ba61.pth'
        opt.bfm_folder = '../deep-3d-face-recon/BFM_2009'
        opt.name = "pretrained" # TODO get model opts globally

        try:
            context.deep_3d_coeffs = get_coeffs_from_image(0, opt, opt.img_folder)
            print("Deep3DCoefficientGenerator: done")
            next_step()
        except Exception as error:
            raise ValueError("Failed getting 3D Face coefficients") from error