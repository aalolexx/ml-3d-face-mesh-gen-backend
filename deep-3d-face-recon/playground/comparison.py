import sys
sys.path.append('../../deep-3d-face-recon')

import numpy as np
from numpy.linalg import norm
from numpy import loadtxt
from scipy.spatial import distance
from coefficient_normalizer import *

alex1 = np.array([loadtxt('face-coeffs/alex1.csv', delimiter=',')]).astype(np.float32)
alex2 = np.array([loadtxt('face-coeffs/alex2.csv', delimiter=',')]).astype(np.float32)
mathis1 = np.array([loadtxt('face-coeffs/mathis1.csv', delimiter=',')]).astype(np.float32)
obama1 = np.array([loadtxt('face-coeffs/obama1.csv', delimiter=',')]).astype(np.float32)
tupac1 = np.array([loadtxt('face-coeffs/tupac1.csv', delimiter=',')]).astype(np.float32)

alex1_norm = normalize_coefficients(np.copy(alex1))
alex2_norm = normalize_coefficients(np.copy(alex2))
mathis1_norm = normalize_coefficients(np.copy(mathis1))
obama1_norm = normalize_coefficients(np.copy(obama1))
tupac1_norm = normalize_coefficients(np.copy(tupac1))

alex1_norm_ne = remove_expression_coefficients(np.copy(alex1_norm))
alex2_norm_ne = remove_expression_coefficients(np.copy(alex2_norm))
mathis1_norm_ne = remove_expression_coefficients(np.copy(mathis1_norm))
obama1_norm_ne = remove_expression_coefficients(np.copy(obama1_norm))
tupac1_norm_ne = remove_expression_coefficients(np.copy(tupac1_norm))

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

print("--------------------- Distances ------------")
print("Cosine Similarity without normalization of coeffs")

print("alex 1 <-> alex 2: "
      + str(cosine_similarity(alex1[0], alex2[0]))
      + " / " + str(cosine_similarity(alex1_norm[0], alex2_norm[0]))
      + " / " + str(cosine_similarity(alex1_norm_ne[0], alex2_norm_ne[0])))
print("alex 1 <-> obama 1: "
      + str(cosine_similarity(alex2[0], obama1[0]))
      + " / " + str(cosine_similarity(alex2_norm[0], obama1_norm[0]))
      + " / "+ str(cosine_similarity(alex2_norm_ne[0], obama1_norm_ne[0])))
print("alex 1 <-> mathis 1: "
      + str(cosine_similarity(alex1[0], mathis1[0]))
      + " / " + str(cosine_similarity(alex1_norm[0], mathis1_norm[0]))
      + " / " + str(cosine_similarity(alex1_norm_ne[0], mathis1_norm_ne[0])))
print("tupac 1 <-> obama 1: "
      + str(cosine_similarity(tupac1[0], obama1[0]))
      + " / " + str(cosine_similarity(tupac1_norm[0], obama1_norm[0]))
      + " / " + str(cosine_similarity(tupac1_norm_ne[0], obama1_norm_ne[0])))