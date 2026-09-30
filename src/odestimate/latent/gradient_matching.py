# core idea: 
# normal gm on observed variables
# for unobserved, pick n_gp_seeds points s_1, ..., s_n_gp_points , that acts as parameters. 
# Interpolate those points using gaussian processes ? (or something fast like that ) 
# optimize gradient matching both with theta and with gp_seeds.