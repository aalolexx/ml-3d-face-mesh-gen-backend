import sys
sys.path.append('../../deep-3d-face-recon')

from models.bfm import ParametricFaceModel
import torch
import numpy as np
from numpy import loadtxt
from util.nvdiffrast import MeshRenderer
from util import util
import trimesh
import matplotlib.pyplot as plt
from coefficient_normalizer import *

# OPTS
bfm_folder = "../BFM_2009"
camera_d = 10.0
focal = 1015.0
center = 112.0
isTrain = False
bfm_model = "BFM_model_front.mat"
z_near = 5.0
z_far = 15.0
use_opengl = True

#device = torch.device('cpu')
device = torch.device(0)
torch.cuda.set_device(device)

print('init done')

alex1_coeffs = np.array([loadtxt('face-coeffs/alex1.csv', delimiter=',')]).astype(np.float32)
alex2_coeffs = np.array([loadtxt('face-coeffs/alex2.csv', delimiter=',')]).astype(np.float32)
alexSmile_coeffs = np.array([loadtxt('face-coeffs/alex_smile.csv', delimiter=',')]).astype(np.float32)
obama1_coeffs = np.array([loadtxt('face-coeffs/obama1.csv', delimiter=',')]).astype(np.float32)
tupac1_coeffs = np.array([loadtxt('face-coeffs/tupac1.csv', delimiter=',')]).astype(np.float32)

print('file read')

alex1_coeffs = torch.from_numpy(alex1_coeffs).to(device)
alex2_coeffs = torch.from_numpy(alex2_coeffs).to(device)
alexSmile_coeffs = torch.from_numpy(alexSmile_coeffs).to(device)
obama1_coeffs = torch.from_numpy(obama1_coeffs).to(device)
tupac1_coeffs = torch.from_numpy(tupac1_coeffs).to(device)

print('tensors created')


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

print('Model setup')

#
# CALCULATE WITH COEFFITIENS
#
def forward_face(coeffs):
    pred_vertex, pred_tex, pred_color, pred_lm = facemodel.compute_for_render(coeffs)
    pred_mask, _, pred_face = renderer(pred_vertex, facemodel.face_buf, feat=pred_color)
    #pred_coeffs_dict = facemodel.split_coeff(output_coeff)
    return pred_vertex, pred_color, pred_mask, pred_face

#
# RENDER AND EXPORT MESH
#
def export_mesh(pred_vertex, pred_color):
    recon_shape = pred_vertex  # get reconstructed shape
    recon_shape[..., -1] = 10 - recon_shape[..., -1]  # from camera space to world space
    recon_shape = recon_shape.cpu().numpy()[0]
    recon_color = pred_color
    recon_color = recon_color.cpu().numpy()[0]
    tri = facemodel.face_buf.cpu().numpy()
    mesh = trimesh.Trimesh(vertices=recon_shape, faces=tri,
                           vertex_colors=np.clip(255. * recon_color, 0, 255).astype(np.uint8), process=False)
    # mesh.export("testobj.obj")

#
# get numpy image from pred face
#


def get_image(coeffs):
    pred_vertex, pred_color, pred_mask, pred_face = forward_face(coeffs)
    pred_face_numpy = pred_face.cpu().numpy()
    output = np.transpose(pred_face_numpy[0], (1, 2, 0))
    output = output.clip(0,1)
    return output
#
# Calulate the faces with slight coeff changes
#

#orig_face = get_image(output_coeff)
#coeffs_alex_1 = torch.clone(output_coeff)


#
# NORMALIZE POSES
#

#id_coeffs = coeffs[:, :80]
#exp_coeffs = coeffs[:, 80: 144]
#tex_coeffs = coeffs[:, 144: 224]
#angles = coeffs[:, 224: 227]
#gammas = coeffs[:, 227: 254]
#translations = coeffs[:, 254:]

alex1_coeffs_norm = torch.clone(alex1_coeffs)
alex2_coeffs_norm = torch.clone(alex2_coeffs)
alexSmile_coeffs_norm = torch.clone(alexSmile_coeffs)
obama1_coeffs_norm = torch.clone(obama1_coeffs)
tupac1_coeffs_norm = torch.clone(tupac1_coeffs)

# angles
alex1_coeffs_norm = normalize_coefficients(alex1_coeffs_norm)
alex2_coeffs_norm = normalize_coefficients(alex2_coeffs_norm)
alexSmile_coeffs_norm = normalize_coefficients(alexSmile_coeffs_norm)
obama1_coeffs_norm = normalize_coefficients(obama1_coeffs_norm)
tupac1_coeffs_norm = normalize_coefficients(tupac1_coeffs_norm)

# NOW ALSO REMOVE EXPRESSIONS

alex1_coeffs_noexp = torch.clone(alex1_coeffs_norm)
alex2_coeffs_noexp = torch.clone(alex2_coeffs_norm)
alexSmile_coeffs_noexp = torch.clone(alexSmile_coeffs_norm)
obama1_coeffs_noexp = torch.clone(obama1_coeffs_norm)
tupac1_coeffs_noexp = torch.clone(tupac1_coeffs_norm)

alex1_coeffs_noexp = remove_expression_coefficients(alex1_coeffs_noexp)
alex2_coeffs_noexp = remove_expression_coefficients(alex2_coeffs_noexp)
alexSmile_coeffs_noexp = remove_expression_coefficients(alexSmile_coeffs_noexp)
obama1_coeffs_noexp = remove_expression_coefficients(obama1_coeffs_noexp)
tupac1_coeffs_noexp = remove_expression_coefficients(tupac1_coeffs_noexp)

#
# RENDER FACES
#

face1 = get_image(alex1_coeffs)
face2 = get_image(alex2_coeffs)
face3 = get_image(alexSmile_coeffs)
face4 = get_image(obama1_coeffs)
face5 = get_image(tupac1_coeffs)
faces = np.hstack((face1, face2, face3, face4, face5))

face1n = get_image(alex1_coeffs_norm)
face2n = get_image(alex2_coeffs_norm)
face3n = get_image(alexSmile_coeffs_norm)
face4n = get_image(obama1_coeffs_norm)
face5n = get_image(tupac1_coeffs_norm)
facesN = np.hstack((face1n, face2n, face3n, face4n, face5n))

face1ne = get_image(alex1_coeffs_noexp)
face2ne = get_image(alex2_coeffs_noexp)
face3ne = get_image(alexSmile_coeffs_noexp)
face4ne = get_image(obama1_coeffs_noexp)
face5ne = get_image(tupac1_coeffs_noexp)
facesNE = np.hstack((face1ne, face2ne, face3ne, face4ne, face5ne))

all_faces = np.vstack((faces, facesN, facesNE))

#
# PREVIEW IMAGE
#
plt.imsave("coeff_x_y_map.png", all_faces)
print("done")
#plt.show()