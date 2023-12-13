"""
OWN CUSTOM SELF WRITTEN FILE
"""
from models.bfm import ParametricFaceModel
import torch
import numpy as np
from util.nvdiffrast import MeshRenderer

# OPTS
# TODO Future: put in conf file
bfm_folder = "../deep-3d-face-recon/BFM_2009"
camera_d = 10.0
focal = 1015.0
center = 112.0
isTrain = False
bfm_model = "BFM_model_front.mat"
z_near = 5.0
z_far = 15.0
use_opengl = True

device = torch.device(0)
torch.cuda.set_device(device)

#
# SET UP FACE MODEL
#
facemodel = ParametricFaceModel(
    bfm_folder=bfm_folder,
    camera_distance=camera_d,
    focal=focal,
    center=center,
    is_train=isTrain,
    default_name=bfm_model
)
facemodel.device = device
facemodel.to(device)

fov = 2 * np.arctan(center / focal) * 180 / np.pi
renderer = MeshRenderer(
    rasterize_fov=fov, znear=z_near, zfar=z_far, rasterize_size=int(2 * center), use_opengl=use_opengl
)


#
# normalizing pose, position, and gamma
#
def normalize_coefficients(bfm_coeffs):
    # angles
    bfm_coeffs[:, 224: 227] = 0
    # translations
    bfm_coeffs[:, 254:] = 0
    # gammas
    bfm_coeffs[:, 227: 254] = 0
    # expression
    bfm_coeffs[:, 80: 144] = 0
    return bfm_coeffs


#
# CALCULATE WITH COEFFITIENS
#
def forward_face(coeffs):
    pred_vertex, pred_tex, pred_color, pred_lm = facemodel.compute_for_render(coeffs)
    pred_mask, _, pred_face = renderer(pred_vertex, facemodel.face_buf, feat=pred_color)
    return pred_vertex, pred_color, pred_mask, pred_face


#
# get numpy image from pred face
#
def get_model_image(coeffs_array, normalize_face=True):
    if normalize_face:
        coeffs_array = normalize_coefficients(coeffs_array)
    pred_vertex, pred_color, pred_mask, pred_face = forward_face(coeffs_array)
    pred_face_numpy = pred_face.cpu().numpy()
    output = np.transpose(pred_face_numpy[0], (1, 2, 0))
    return output.clip(0, 1)
