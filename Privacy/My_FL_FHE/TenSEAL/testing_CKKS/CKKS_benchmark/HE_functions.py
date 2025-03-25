import tenseal as ts
import numpy as np

def Ckks_init(degree, coeff):
    
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree = degree,
        coeff_mod_bit_sizes = coeff,
    )
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


def flatten_params(weights, biases):
    original_bias_shapes = [b.shape for b in biases]
    flattened_biases = np.concatenate([b.flatten() for b in biases])

    original_weights_shapes = [w.shape for w in weights]
    flattened_weights = np.concatenate([w.flatten() for w in weights])
    # print("Weights:", flattened_weights, " biases :", flattened_biases)

    original_shapes = (original_weights_shapes, original_bias_shapes)

    return original_shapes, flattened_weights, flattened_biases

def reverse_params(original_shapes_combined, flat_weights, flat_biases):
    
    """
    Reverses the flattening of the weights and biases, returning them to their original shape.
    """
    original_weights_shapes, original_bias_shapes = original_shapes_combined
    # print(original_weights_shapes, original_bias_shapes)

    # Reverse the weights
    weights_reversed = []
    index = 0
    for weight_shape in original_weights_shapes:
        num_elements = np.prod(weight_shape)
        weight_matrix = flat_weights[index:index + num_elements].reshape(weight_shape)
        weights_reversed.append(weight_matrix)
        index += num_elements

    # Reverse the biases
    biases_reversed = []
    bias_index = 0
    for bias_shape in original_bias_shapes:
        num_elements = np.prod(bias_shape)
        bias_vector = flat_biases[bias_index:bias_index + num_elements].reshape(bias_shape)
        biases_reversed.append(bias_vector)
        bias_index += num_elements

    return weights_reversed, biases_reversed