# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 20:40:01 2020

@author: Thomas Lokken 4449282
"""

import scipy as sp
import matplotlib.pyplot as plt
import scipy.optimize as opt
import matplotlib.pylab as pl


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
height = 1 


# =============================================================================
# Iteration variables
# =============================================================================
n_characteristics = 10                                      #number of points at the exit diameter
nozzle_exit = sp.zeros(shape=(24,2))
shock_occurs = False
current_element_index  = 0
while_index = 0
range_x = 10
steps_x = 10000
x = sp.linspace(0, range_x, steps_x)



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

def calc_positive_slope(mu, phi):
    return sp.tan(degrees_to_radians(phi + mu))

def calc_negative_slope(mu, phi):
    return sp.tan(degrees_to_radians(phi - mu))

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
# first expansion parameters
gamma_minus_mach_array_ABC = sp.zeros([characteristics])
gamma_minus_nu_array_ABC = sp.zeros([characteristics])
gamma_minus_phi_array_ABC = sp.zeros([characteristics])

# first intersection points complex waves
x_intersections_BCE = sp.zeros([characteristics, characteristics])
y_intersections_BCE = sp.zeros([characteristics, characteristics])
x_intersections_DFG = sp.zeros([characteristics, characteristics])
y_intersections_DFG = sp.zeros([characteristics, characteristics])

# first intersection points complex waves
complex_wave_nu_BCE = sp.zeros([characteristics, characteristics])
complex_wave_phi_BCE = sp.zeros([characteristics, characteristics])
complex_wave_alpha_BCE = sp.zeros([characteristics, characteristics])
complex_wave_mu_BCE = sp.zeros([characteristics, characteristics])
complex_wave_nu_DFG = sp.zeros([characteristics, characteristics])
complex_wave_phi_DFG = sp.zeros([characteristics, characteristics])
complex_wave_alpha_DFG = sp.zeros([characteristics, characteristics])
complex_wave_mu_DFG = sp.zeros([characteristics, characteristics])



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
        complex_wave_alpha_BCE[i][j] = calc_positive_slope(complex_wave_mu_BCE[i][j], complex_wave_phi_BCE[i][j])

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
# Finding characteristic intersections ABC
# =============================================================================
gammas = sp.zeros([characteristics,len(x)])
jet_boundary = height + sp.tan(degrees_to_radians(calc_alpha(calc_mu_from_Mach(mach_B), phi_B))) * x
pl.figure()
for i in sp.arange(0, characteristics, 1):
    for j in sp.arange(i, characteristics, 1):
        
        positive_slope = 0
        negative_slope = 0
        if i == 0:
            #incoming_characteristic = height - sp.tan(calc_alpha(degrees_to_radians(calc_mu_from_Mach(gamma_minus_mach_array_ABC[j])), degrees_to_radians(gamma_minus_phi_array_ABC[j]))) * x
            incoming_characteristic = height + calc_negative_slope(calc_mu_from_Mach(gamma_minus_mach_array_ABC[j]), gamma_minus_phi_array_ABC[j]) * x
            negative_slope = calc_negative_slope(calc_mu_from_Mach(gamma_minus_mach_array_ABC[j]), gamma_minus_phi_array_ABC[j])
            if j == 0:
                reflected_characteristic = 0 * x
            else: 
                prev_x = x_intersections_BCE[i][j-1]
                positive_slope =  calc_positive_slope(complex_wave_nu_BCE[i][j], complex_wave_phi_BCE[i][j])
                diff_slopes = y_intersections_BCE[i][j-1] - positive_slope * x_intersections_BCE[i][j-1]
                reflected_characteristic = positive_slope * x + diff_slopes 
        else:
            negative_slope = calc_negative_slope(complex_wave_nu_BCE[i][j], complex_wave_phi_BCE[i][j])
            if j == i:
                positive_slope = 0
            else:
                positive_slope = calc_positive_slope(complex_wave_nu_BCE[i][j], complex_wave_phi_BCE[i][j])
            incoming_characteristic = y_intersections_BCE[i-1][j] + negative_slope * x
            reflected_characteristic = y_intersections_BCE[i][j-1] + positive_slope * x
            
        y_idx = sp.argwhere(sp.diff(sp.sign(reflected_characteristic - incoming_characteristic))).flatten()
        y_intersections_BCE[i][j] = incoming_characteristic[y_idx]
        if i > 0:
            x_intersections_BCE[i][j] = x_intersections_BCE[i-1][j] + (y_intersections_BCE[i][j] - y_intersections_BCE[i-1][j]) / negative_slope
        else: 
            x_intersections_BCE[i][j] = x_intersections_BCE[i][j] + x[y_idx]

# =============================================================================
#       Finding the arrays to plot the characteristics of ABC
# =============================================================================
        x_idx = sp.where(x >= x_intersections_BCE[i][j])
        if i ==0 : #first wave intersection
            gammas[j][:y_idx[0]]= 1 + negative_slope * x[:y_idx[0]] 
            prev_x_idx_reflected = sp.where(x > x_intersections_BCE[i][j-1])
            if j > 0:
                gammas[i][prev_x_idx_reflected[0][0]:x_idx[0][0] + 1]= y_intersections_BCE[i][j-1] + positive_slope * (x[prev_x_idx_reflected[0][0]:x_idx[0][0] + 1] - x[prev_x_idx_reflected[0][0]])

        elif (j == i): #intersection with x-axis
            prev_x_idx = sp.where(x >= x_intersections_BCE[i-1][j])
            gammas[j][prev_x_idx[0][0]:x_idx[0][0] +1]= y_intersections_BCE[i-1][j] + negative_slope * (x[prev_x_idx[0][0]:x_idx[0][0] +1] - x[prev_x_idx[0][0]])

        else:
            prev_x_idx = sp.where(x >= x_intersections_BCE[i-1][j])
            idx = sp.where(x >= x_intersections_BCE[i][j])
            prev_x_idx_reflected = sp.where(x > x_intersections_BCE[i][j-1])
            gammas[j][prev_x_idx[0][0]:x_idx[0][0] +1]= y_intersections_BCE[i-1][j] + negative_slope * x[prev_x_idx[0][0]:x_idx[0][0] + 1] - negative_slope * x[prev_x_idx[0][0]]
            gammas[i][prev_x_idx_reflected[0][0]:x_idx[0][0] + 1]= y_intersections_BCE[i][j-1] + (y_intersections_BCE[i][j] - y_intersections_BCE[i][j-1]) / (x_intersections_BCE[i][j] - x_intersections_BCE[i][j-1])  * (x[prev_x_idx_reflected[0][0]:x_idx[0][0] + 1] - x[prev_x_idx_reflected[0][0]])

        if j == characteristics - 1:
            gammas[i][x_idx[0][0]:]= y_intersections_BCE[i][j] + complex_wave_alpha_BCE[i][j] * (x[x_idx[0][0]:]- x[x_idx[0][0]])

# =============================================================================
#           Plotting the characteristics of ABC
# =============================================================================
            if i == 0 or i == characteristics -1:
                pl.plot(x, gammas[i], '-', color="blue")
            else:
                pl.plot(x, gammas[i], '--', color="blue")
pl.plot(x, jet_boundary, '--', color="red")
intersections_masked = pl.ma.masked_where((x_intersections_BCE<0.1)&(x_intersections_BCE>-0.1), y_intersections_BCE)
pl.plot(x_intersections_BCE,intersections_masked, '.', color="black")


# =============================================================================
# Finding characteristic intersections DFG
# =============================================================================
for i in sp.arange(0, characteristics, 1):
    for j in sp.arange(i, characteristics, 1):
        
        positive_slope = 0
        negative_slope = 0
        if i == 0:
            #incoming_characteristic = height - sp.tan(calc_alpha(degrees_to_radians(calc_mu_from_Mach(gamma_minus_mach_array_ABC[j])), degrees_to_radians(gamma_minus_phi_array_ABC[j]))) * x
            print(y_intersections_BCE[i][-1])
            incoming_characteristic = y_intersections_BCE[i][-1] + complex_wave_alpha_BCE[i][-1] * x
            negative_slope = calc_negative_slope(complex_wave_mu_DFG[i][j], complex_wave_phi_DFG[i][j])
            if j == i:
                reflected_characteristic = 0 * x
            else: 
                prev_x = x_intersections_DFG[i][j-1]
                positive_slope =  calc_positive_slope(complex_wave_nu_DFG[i][j], complex_wave_phi_DFG[i][j])
                diff_slopes = y_intersections_DFG[i][j-1] - positive_slope * x_intersections_DFG[i][j-1]
                reflected_characteristic = positive_slope * x + diff_slopes 
        else:
            negative_slope = calc_negative_slope(complex_wave_nu_DFG[i][j], complex_wave_phi_DFG[i][j])
            if j == i:
                positive_slope = 0
            else:
                positive_slope = calc_positive_slope(complex_wave_nu_DFG[i][j], complex_wave_phi_DFG[i][j])
            incoming_characteristic = y_intersections_DFG[i-1][j] + negative_slope * x
            reflected_characteristic = y_intersections_DFG[i][j-1] + positive_slope * x
            
        y_idx = sp.argwhere(sp.diff(sp.sign(reflected_characteristic - incoming_characteristic))).flatten()
        y_intersections_DFG[i][j] = incoming_characteristic[y_idx]
        if i > 0:
            x_intersections_DFG[i][j] = x_intersections_DFG[i-1][j] + (y_intersections_DFG[i][j] - y_intersections_DFG[i-1][j]) / negative_slope
        else: 
            x_intersections_DFG[i][j] = x_intersections_DFG[i][j] + x[y_idx]

# =============================================================================
#       Finding the arrays to plot the characteristics DFG
# =============================================================================
        x_idx = sp.where(x >= x_intersections_DFG[i][j])
        if i ==0 : #first wave intersection
            gammas[j][:y_idx[0]]= 1 + negative_slope * x[:y_idx[0]] 
            prev_x_idx_reflected = sp.where(x > x_intersections_DFG[i][j-1])
            if j > 0:
                gammas[i][prev_x_idx_reflected[0][0]:x_idx[0][0] + 1]= y_intersections_DFG[i][j-1] + positive_slope * (x[prev_x_idx_reflected[0][0]:x_idx[0][0] + 1] - x[prev_x_idx_reflected[0][0]])

        elif (j == i): #intersection with x-axis
            prev_x_idx = sp.where(x >= x_intersections_DFG[i-1][j])
            gammas[j][prev_x_idx[0][0]:x_idx[0][0] +1]= y_intersections_DFG[i-1][j] + negative_slope * (x[prev_x_idx[0][0]:x_idx[0][0] +1] - x[prev_x_idx[0][0]])

        else:
            prev_x_idx = sp.where(x >= x_intersections_DFG[i-1][j])
            idx = sp.where(x >= x_intersections_DFG[i][j])
            prev_x_idx_reflected = sp.where(x > x_intersections_DFG[i][j-1])
            gammas[j][prev_x_idx[0][0]:x_idx[0][0] +1]= y_intersections_DFG[i-1][j] + negative_slope * x[prev_x_idx[0][0]:x_idx[0][0] + 1] - negative_slope * x[prev_x_idx[0][0]]
            gammas[i][prev_x_idx_reflected[0][0]:x_idx[0][0] + 1]= y_intersections_DFG[i][j-1] + (y_intersections_DFG[i][j] - y_intersections_DFG[i][j-1]) / (x_intersections_DFG[i][j] - x_intersections_DFG[i][j-1])  * (x[prev_x_idx_reflected[0][0]:x_idx[0][0] + 1] - x[prev_x_idx_reflected[0][0]])

        if j == characteristics - 1:
            gammas[i][x_idx[0][0]:]= y_intersections_DFG[i][j] + complex_wave_alpha_DFG[i][j] * (x[x_idx[0][0]:]- x[x_idx[0][0]])

# =============================================================================
#       Plot characteristics for DFG
# =============================================================================

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