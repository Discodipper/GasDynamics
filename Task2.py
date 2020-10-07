# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 20:40:01 2020

@author: Thomas Lokken 4449282
"""

import scipy as sp
import matplotlib.pyplot as plt
import scipy.optimize as opt

# =============================================================================
# Constants
# =============================================================================
gamma = 7/5

# =============================================================================
# Initial variables
# =============================================================================
mach_exit = 2                               #-
exit_pressore_over_ambient_pressure = 2     #-
phi_exit = 0                                #degrees
ambient_pressure = 1225                     #N/m


# =============================================================================
# Iteration variables
# =============================================================================
n_characteristics = 10                                      #number of points at the exit diameter
nozzle_exit = sp.zeros(shape=(24,2))
shock_occurs = False
current_element_index  = 0
while_index = 0
x = sp.linspace(0, 8, 1000)



def degrees_to_radians(angle):
    return angle / 180 * sp.pi

def radians_to_degrees(angle):
    return angle * 180 / sp.pi

# calculate angle mu from mach number
def calc_mu_from_Mach(mach):
    return radians_to_degrees(sp.arcsin(1/mach))

# calculate angle mu from mach number
def calc_Mach_from_mu(mu):
    return 1/sp.sin(degrees_to_radians(mu))

# calculate the Prandtl-meyer angle
def calc_Prandtl_Meyer_from_Mach(mach):
    return radians_to_degrees(sp.sqrt((gamma+1)/(gamma-1)) * sp.arctan(sp.sqrt((gamma-1)/(gamma+1) * (mach**2-1))) - sp.arctan(sp.sqrt(mach**2 -1)))

def calc_alpha(mu, phi):
    return mu - phi

def calc_forward_phi_gamma_plus(nu_backward, phi_backward, nu_forward):
    return - (nu_backward - phi_backward - nu_forward)

def find_mach_for_prandtl_meyer(mach, nu):
    return calc_Prandtl_Meyer_from_Mach(mach) - nu
    
def find_mu_from_nu(nu, mach_head, mach_tail, tolerance):
    mach_left = mach_head
    mach_right = mach_tail
    while (sp.absolute(mach_left - mach_right) >= tolerance):
        output_mach = (mach_left + mach_right)/2.0
        step = find_mach_for_prandtl_meyer(output_mach, nu)
        if sp.absolute(step) < sp.absolute(tolerance): 
            break
        if step * find_mach_for_prandtl_meyer(mach_left, nu) < tolerance:
            mach_right = output_mach
        else:
            mach_left = output_mach
        #het gaat hier mis
    mu = calc_mu_from_Mach(output_mach)
    return mu, output_mach

def bisection_step_function(mach, alpha, backward_phi, backward_nu):
    product = calc_Prandtl_Meyer_from_Mach(mach) - calc_mu_from_Mach(mach) + alpha + backward_phi - backward_nu
    return product

def bisection_method(mach_head, mach_tail, tolerance, alpha, backward_phi, backward_nu):
    mach_left = mach_head
    mach_right = mach_tail
    while (sp.absolute(mach_left - mach_right) >= tolerance):
        output_mach = (mach_left + mach_right)/2.0
        step = bisection_step_function(output_mach, alpha, backward_phi, backward_nu)
        if sp.absolute(step) < sp.absolute(tolerance): 
            break
        if step * bisection_step_function(mach_left, alpha, backward_phi, backward_nu) < tolerance:
            mach_right = output_mach
        else:
            mach_left = output_mach
    mu = calc_mu_from_Mach(output_mach)
    phi = mu - alpha
    nu = calc_Prandtl_Meyer_from_Mach(output_mach)  
    return output_mach, phi, nu

def translate_angle_to_gradient(angle):
     return sp.arccos(angle)

# =============================================================================
# Iteration parameters
# =============================================================================
characteristics = 10
gamma_minus_mach_array_ABC = sp.zeros([characteristics])
gamma_minus_nu_array_ABC = sp.zeros([characteristics])
gamma_minus_phi_array_ABC = sp.zeros([characteristics])
x_intersections_ABC = sp.zeros([characteristics, characteristics])
y_intersections_ABC = sp.zeros([characteristics, characteristics])
idx_intersections_ABC = sp.zeros([characteristics, characteristics])
complex_wave_nu_BCE = sp.zeros([characteristics, characteristics])
complex_wave_phi_BCE = sp.zeros([characteristics, characteristics])
complex_wave_alpha_BCE = sp.zeros([characteristics, characteristics])
complex_wave_mu_BCE = sp.zeros([characteristics, characteristics])
complex_wave_nu_DFG = sp.zeros([characteristics, characteristics])
complex_wave_phi_DFG = sp.zeros([characteristics, characteristics])



# =============================================================================
# mu_array[0] = calc_mu_from_Mach(mach_exit)
# =============================================================================
error_tolerance = 0.0001

# =============================================================================
# Necessary variables to perform iteration
# =============================================================================
mach_B = sp.sqrt(2 / (gamma-1) * (2**((gamma - 1) / gamma) * (1 + (gamma-1) / 2 * mach_exit**2) -1))
phi_B = calc_forward_phi_gamma_plus(calc_Prandtl_Meyer_from_Mach(mach_exit), phi_exit, calc_Prandtl_Meyer_from_Mach(mach_B))
alpha_0 =  calc_alpha(calc_mu_from_Mach(mach_exit), phi_exit)
delta_alpha = calc_alpha(calc_mu_from_Mach(mach_B), phi_B) - alpha_0
alpha_steps = enumerate(sp.arange(
        alpha_0,
        alpha_0 + delta_alpha + delta_alpha / (characteristics),
        delta_alpha / (characteristics - 1)
    ))

for j, alpha in alpha_steps:
    if (j == 0):
        gamma_minus_mach_array_ABC[j] = mach_exit
        gamma_minus_phi_array_ABC[j] = phi_exit
        gamma_minus_nu_array_ABC[j] = calc_Prandtl_Meyer_from_Mach(mach_exit)
    else :
        gamma_minus_mach_array_ABC[j], gamma_minus_phi_array_ABC[j], gamma_minus_nu_array_ABC[j] = bisection_method(
            gamma_minus_mach_array_ABC[j - 1] if j > 0 else mach_exit,
            mach_B, 
            error_tolerance, 
            alpha, 
            gamma_minus_phi_array_ABC[j - 1] if j > 0 else phi_exit, 
            calc_Prandtl_Meyer_from_Mach(gamma_minus_mach_array_ABC[j - 1] if j > 0 else mach_exit)
    )
    
    for i in sp.arange(0, j + 1, 1):
        if i == j:
            complex_wave_nu_BCE[i][j] = gamma_minus_nu_array_ABC[j] + gamma_minus_phi_array_ABC[j]
        elif i == 0:
            complex_wave_nu_BCE[i][j] = 0.5*(complex_wave_nu_BCE[i][j-1] + gamma_minus_nu_array_ABC[j]) + 0.5 * (gamma_minus_phi_array_ABC[j] - complex_wave_phi_BCE[i][j-1])
            complex_wave_phi_BCE[i][j] = 0.5*(complex_wave_phi_BCE[i][j-1] + gamma_minus_phi_array_ABC[j]) + 0.5 * (gamma_minus_nu_array_ABC[j] - complex_wave_nu_BCE[i][j-1])
        else:
            complex_wave_nu_BCE[i][j] = 0.5*(complex_wave_nu_BCE[i][j-1] + complex_wave_nu_BCE[i-1][j]) + 0.5 * (complex_wave_phi_BCE[i-1][j] - complex_wave_phi_BCE[i][j-1])
            complex_wave_phi_BCE[i][j] = 0.5*(complex_wave_phi_BCE[i][j-1] + complex_wave_phi_BCE[i-1][j]) + 0.5 * (complex_wave_nu_BCE[i-1][j] - complex_wave_nu_BCE[i][j-1])
        complex_wave_mu_BCE[i][j], mach = find_mu_from_nu(complex_wave_nu_BCE[i][j], mach_exit, mach_B+ 10, error_tolerance)
        complex_wave_alpha_BCE[i][j] = calc_alpha(complex_wave_mu_BCE[i][j], complex_wave_phi_BCE[i][j])

for j in sp.arange(0, characteristics, 1):
    for i in sp.arange(0, j+1, 1):
        if i == j:
            complex_wave_nu_DFG[i][j] =  complex_wave_nu_BCE[i][-1] + complex_wave_phi_BCE[i][-1]
        elif i == 0 :
            complex_wave_nu_DFG[i][j] = 0.5 * (complex_wave_nu_BCE[i][-1] + complex_wave_nu_DFG[i][j]) + 0.5 * (gamma_minus_phi_array_ABC[j] - complex_wave_phi_BCE[i][-1])
            complex_wave_phi_DFG[i][j] = 0.5 * (complex_wave_phi_BCE[i][-1] + complex_wave_phi_DFG[i][j]) + 0.5 * (gamma_minus_nu_array_ABC[j] - complex_wave_nu_BCE[i][-1])
        else:
            complex_wave_nu_DFG[i][j] = 0.5 * (complex_wave_nu_DFG[i][j-1] + complex_wave_nu_DFG[i-1][j]) + 0.5 * (complex_wave_phi_DFG[i-1][j] - complex_wave_phi_DFG[i][j-1])
            complex_wave_phi_DFG[i][j] = 0.5 * (complex_wave_phi_DFG[i][j-1] + complex_wave_phi_DFG[i-1][j]) + 0.5 * (complex_wave_nu_DFG[i-1][j] - complex_wave_nu_DFG[i][j-1])

# =============================================================================
# Finding characteristic intersections
# =============================================================================
g = 0 * x
for i in sp.arange(0, characteristics, 1):
    for j in sp.arange(i, characteristics, 1):
        
        incoming_characteristic = 1 - sp.tan(calc_alpha(degrees_to_radians(calc_mu_from_Mach(gamma_minus_mach_array_ABC[j])), degrees_to_radians(gamma_minus_phi_array_ABC[j]))) * x
        if j == 0:
            reflected_characteristic = 0 * x
            plt.plot(x, incoming_characteristic)
        elif j == i:
            reflected_characteristic = sp.tan(degrees_to_radians(complex_wave_alpha_BCE[i][j-1])) * x - y_intersections_ABC[i][j-1]
        else:
            print(complex_wave_alpha_BCE[i][j-1])
            reflected_characteristic = sp.tan(degrees_to_radians(complex_wave_alpha_BCE[i][j-1])) * x - y_intersections_ABC[i][j-1]
            incoming_characteristic = 1 - sp.tan(degrees_to_radians(complex_wave_alpha_BCE[i][j])) * x - y_intersections_ABC[i-1][j]
        idx = sp.argwhere(sp.diff(sp.sign(reflected_characteristic - incoming_characteristic))).flatten()
        print(x[idx])
        x_intersections_ABC[i][j] = x[idx]
        y_intersections_ABC[i][j] = y_intersections_ABC[i][j-1] +  sp.tan(degrees_to_radians(complex_wave_alpha_BCE[i][j-1])) * (x_intersections_ABC[i][j] - x_intersections_ABC[i][j-1])
# =============================================================================
#         plt.plot(x_intersections_ABC[0], y_intersections_ABC[0], marker='o')
# =============================================================================
# =============================================================================
#             x_intersections_ABC[i][j] = x[idx]
#             idx_intersections_ABC = reflected_characteristic[idx]
# =============================================================================
# =============================================================================
#             x_intersection = opt(
#                 f,
#                 x_intersections_ABC[i][j-1], 
#                 x[499]
#             )
#             print(x_intersection)
# =============================================================================
            # x_intersection = x[sp.argwhere(sp.diff(sp.sign(f - g))).flatten()]

        if j > 0 and j < characteristics - 1:
            g = 1 - sp.arctan(complex_wave_alpha_BCE[i][j+1]) * x
        


# =============================================================================
# Plotting the characteristics
# =============================================================================

y = sp.tan(alpha)





# =============================================================================
# Maybe comes in handy later
# =============================================================================
# =============================================================================
# while shock_occurs ==  False:
#     nu_0 = calc_Prandtl_Meyer_from_Mach(mach_exit)
#     for i in sp.arange(0, (n_points - while_index)):
#         if (current_element_index < n_points) :
#             nozzle_exit[current_element_index, 1] = nu_0
#             nozzle_exit[current_element_index, 0] = phi_exit
#         else :
#             backward_upper_idx = current_element_index - (n_points - while_index) - 1 
#             backward_lower_idx = current_element_index - (n_points - while_index)
#             nozzle_exit[current_element_index, 0] = 0.5* (nozzle_exit[backward_lower_idx, 0] + nozzle_exit[backward_upper_idx, 0]) + 0.5 * (nozzle_exit[backward_upper_idx, 1] - nozzle_exit[backward_lower_idx, 1])
#             nozzle_exit[current_element_index, 1] = 0.5* (nozzle_exit[backward_lower_idx, 1] + nozzle_exit[backward_upper_idx, 1]) + 0.5 * (nozzle_exit[backward_upper_idx, 0] - nozzle_exit[backward_lower_idx, 0])
#         current_element_index += 1
#     
#     while_index += 1
#     
#     if (while_index == n_points):
#         shock_occurs = True
#         
# =============================================================================