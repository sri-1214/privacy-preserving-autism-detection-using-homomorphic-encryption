import tenseal as ts
import numpy as np

# ======================
# CREATE CKKS CONTEXT
# ======================
def create_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context


# ======================
# ENCRYPT VECTOR
# ======================
def encrypt_vector(context, vector):
    return ts.ckks_vector(context, vector.tolist())


# ======================
# DECRYPT VECTOR
# ======================
def decrypt_vector(enc_vec):
    return np.array(enc_vec.decrypt())


# ======================
# ENCRYPTED FORWARD
# ======================
def encrypted_forward(enc_x, w, b):
    return enc_x.matmul(w.T) + b