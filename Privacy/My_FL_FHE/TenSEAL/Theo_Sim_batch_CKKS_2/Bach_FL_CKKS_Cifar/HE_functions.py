import tenseal as ts
import numpy as np

def Ckks_init(degree, coeff, pow = None):
    
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree = degree,
        coeff_mod_bit_sizes = coeff,
    )
    if pow == None:
        pow = min(coeff)
    context.global_scale = 2**pow

    context.generate_galois_keys()

    return context



def Encrypt_model(context, weights :list , bias :list):

    encrypted_weights = ts.ckks_vector(context, weights)  
    encrypted_bias = ts.ckks_vector(context, bias) 

    return encrypted_weights, encrypted_bias



def Dencrypt_model(encrypted_weight, encrypted_bias):

    decrypted_weight = encrypted_weight.decrypt() 
    decrypted_bias = encrypted_bias.decrypt()

    decrypted_weight = np.array(decrypted_weight)
    decrypted_bias = np.array(decrypted_bias)

    return decrypted_weight, decrypted_bias