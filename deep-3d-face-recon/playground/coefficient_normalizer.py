#normalizing pose, position, and gamma
def normalize_coefficients(bfm_coeffs):
    # angles
    bfm_coeffs[:, 224: 227] = 0
    # translations
    bfm_coeffs[:, 254:] = 0
    # gammas
    bfm_coeffs[:, 227: 254] = 0
    return bfm_coeffs

def remove_expression_coefficients(bfm_coeffs):
    bfm_coeffs[:, 80: 144] = 0
    return bfm_coeffs