configurations = [
    # poly_modulus_degree = 2048 (Max sum ≈ 54 bits)
    (2048, [20, 20, 14]),
    (2048, [20, 20]),
    (2048, [18, 18]),
    (2048, [16, 16]),

    # poly_modulus_degree = 4096 (Max sum ≈ 109 bits)
    (4096, [30, 30, 30, 19]),  # Smaller primes for more levels
    (4096, [40, 40, 29]),      # Balanced precision
    (4096, [50, 50]),          # Fewer levels, higher precision
    (4096, [40, 20, 40]),
    (4096, [30, 20, 30]),
    (4096, [20, 20, 20]),
    (4096, [19, 19, 19]),
    (4096, [18, 18, 18]),
    (4096, [18, 18]),
    (4096, [17, 17]),

    # poly_modulus_degree = 8192 (Max sum ≈ 218 bits)
    (8192, [40, 21, 21, 21, 21, 21, 21, 40]),
    (8192, [40, 40, 40, 40, 40]),          
    (8192, [60, 50, 60]),               
    (8192, [40, 20, 40]),
    (8192, [20, 20, 20]),
    (8192, [17, 17]),

    # poly_modulus_degree = 16384 (Max sum ≈ 438 bits)
    (16384, [54, 54, 54, 54, 54, 54, 54, 54]),
    (16384, [60, 60, 60, 60, 60, 60, 60]),
    (16384, [60, 60, 60, 60, 60, 60]),
    (16384, [60, 54, 54, 60]),     
    (16384, [60, 60]),         


    # poly_modulus_degree = 32768 (Max sum ≈ 881 bits)
    (32768, [60, 60, 60, 60, 60, 60, 60, 60, 60, 60]),  
    (32768, [60, 60, 60, 60, 60]),        
    (32768, [60, 60, 60, 60]),            
    (32768, [60, 60, 60]), 
    (32768, [60, 60]),                      
]

# bits_scale = 26

# # Create TenSEAL context
# context = ts.context(
#     ts.SCHEME_TYPE.CKKS,
#     poly_modulus_degree=8192,
#     coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
# )

# # set the scale
# context.global_scale = pow(2, bits_scale)