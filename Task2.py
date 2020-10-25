# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 20:40:01 2020

@author: Thomas Lokken 4449282
"""

import scipy as sp
import matplotlib.pyplot as plt
import scipy.optimize as opt
import matplotlib.pylab as pl
import matplotlib.patches as ptch


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
height = 1                                  #m
P_tot = 1                                   #N/m^2


# =============================================================================
# Iteration variables
# =============================================================================
n_characteristics = 10                      #number of points at the exit diameter
range_x = 14                                #limit of x
steps_x = 10000                             #steps covering domain
x = sp.linspace(0, range_x, steps_x)        #all the x values for the steps


def degrees_to_radians(angle):
    return angle / 180 * sp.pi

def radians_to_degrees(angle):
    return angle * 180 / sp.pi

# calculate angle mu from mach number
def calc_mu_from_Mach(mach):
    return radians_to_degrees(sp.arcsin(1/mach))

# calculate the Prandtl-meyer angle
def calc_Prandtl_Meyer_from_Mach(mach):
    return radians_to_degrees(sp.sqrt((gamma+1)/(gamma-1)) * sp.arctan(sp.sqrt((gamma-1)/(gamma+1) * (mach**2-1))) - sp.arctan(sp.sqrt(mach**2 -1)))

def calc_alpha(mu, phi):
    return phi - mu

def calc_positive_slope(mu, phi):
    return sp.tan(degrees_to_radians(phi + mu))

def calc_negative_slope(mu, phi):
    return sp.tan(degrees_to_radians(phi - mu))

def calc_forward_phi_gamma_plus(nu_backward, phi_backward, nu_forward):
    return nu_forward + phi_backward - nu_backward

def find_mach_for_prandtl_meyer(mach, nu):
    return calc_Prandtl_Meyer_from_Mach(mach) - nu

def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in values]

    
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
    return - calc_Prandtl_Meyer_from_Mach(mach) + calc_mu_from_Mach(mach) + alpha - backward_phi + backward_nu

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
    phi = mu + alpha
    nu = calc_Prandtl_Meyer_from_Mach(output_mach)  
    return output_mach, phi, nu

def make_linear_function(a, b, x, correction = 0 ):
    return a * (x - correction) + b

def pressure_of_total_pressure(mach):
    return (1 + (gamma-1)/2 * mach**2)**(-gamma/(gamma-1)) * P_tot

def pressure_distribution_centre(press_dist, x_crossings):
    for i in sp.arange(n_characteristics):
        x_pos = x_intersections_BCE[i][i]
        press_dist = sp.hstack((press_dist, pressure_of_total_pressure(complex_wave_mach_BCE[i][i])))
        x_crossings = sp.hstack((x_crossings, x_pos))
    for i in sp.arange(n_characteristics):
        x_pos = x_intersections_HIJ[i][i]
        press_dist = sp.hstack((press_dist, pressure_of_total_pressure(complex_wave_mach_HIJ[i][i])))
        x_crossings = sp.hstack((x_crossings, x_pos))
    return press_dist, x_crossings

# =============================================================================
# Iteration parameters
# =============================================================================
# first expansion parameters
gamma_minus_mach_array_ABC = sp.zeros([n_characteristics])
gamma_minus_nu_array_ABC = sp.zeros([n_characteristics])
gamma_minus_phi_array_ABC = sp.zeros([n_characteristics])

# first intersection points complex waves
x_intersections_BCE = sp.zeros([n_characteristics, n_characteristics])
y_intersections_BCE = sp.zeros([n_characteristics, n_characteristics])

x_intersections_DFG = sp.zeros([n_characteristics, n_characteristics])
y_intersections_DFG = sp.zeros([n_characteristics, n_characteristics])

x_intersections_HIJ = sp.zeros([n_characteristics, n_characteristics])
y_intersections_HIJ = sp.zeros([n_characteristics, n_characteristics])

# first intersection points complex waves
complex_wave_nu_BCE = sp.zeros([n_characteristics, n_characteristics])
complex_wave_phi_BCE = sp.zeros([n_characteristics, n_characteristics])
complex_wave_alpha_BCE = sp.zeros([n_characteristics, n_characteristics])
complex_wave_mu_BCE = sp.zeros([n_characteristics, n_characteristics])
complex_wave_mach_BCE = sp.zeros([n_characteristics, n_characteristics])

complex_wave_nu_DFG = sp.zeros([n_characteristics, n_characteristics])
complex_wave_phi_DFG = sp.zeros([n_characteristics, n_characteristics])
complex_wave_alpha_DFG = sp.zeros([n_characteristics, n_characteristics])
complex_wave_mu_DFG = sp.zeros([n_characteristics, n_characteristics])
complex_wave_mach_DFG = sp.zeros([n_characteristics, n_characteristics])

complex_wave_nu_HIJ = sp.zeros([n_characteristics, n_characteristics])
complex_wave_phi_HIJ = sp.zeros([n_characteristics, n_characteristics])
complex_wave_alpha_HIJ = sp.zeros([n_characteristics, n_characteristics])
complex_wave_mu_HIJ = sp.zeros([n_characteristics, n_characteristics])
complex_wave_mach_HIJ = sp.zeros([n_characteristics, n_characteristics])

error_tolerance = 0.0001

# =============================================================================
# Necessary calculations for characteristic points
# =============================================================================
mach_B = sp.sqrt(2 / (gamma-1) * (2**((gamma - 1) / gamma) * (1 + (gamma-1) / 2 * mach_exit**2) -1))
phi_B = calc_forward_phi_gamma_plus(calc_Prandtl_Meyer_from_Mach(mach_exit), phi_exit, calc_Prandtl_Meyer_from_Mach(mach_B))
alpha_0 =  calc_alpha(calc_mu_from_Mach(mach_exit), phi_exit)
delta_alpha = calc_alpha(calc_mu_from_Mach(mach_B), phi_B) - alpha_0

# Apply an enumeration over the list of alpha steps to keep track of the relevant index
alpha_steps = enumerate(sp.arange(
        alpha_0,
        alpha_0 + delta_alpha + delta_alpha / (n_characteristics),
        delta_alpha / (n_characteristics - 1)
    ))

# =============================================================================
# This section of code is dedicated to finding all the characteristic variables 
# to the three different domains of the jet area. 
# =============================================================================
for j, alpha in alpha_steps: #Iterate over the alpha steps with output angle and index
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

    for i in sp.arange(0, j + 1, 1): #crossing characteristics
        if i == j:
            complex_wave_nu_BCE[i][j] = gamma_minus_nu_array_ABC[j] + gamma_minus_phi_array_ABC[j]
        elif i == 0:
            complex_wave_nu_BCE[i][j] = 0.5*(complex_wave_nu_BCE[i][j-1] + gamma_minus_nu_array_ABC[j]) + 0.5 * (gamma_minus_phi_array_ABC[j] - complex_wave_phi_BCE[i][j-1])
            complex_wave_phi_BCE[i][j] = 0.5*(complex_wave_phi_BCE[i][j-1] + gamma_minus_phi_array_ABC[j]) + 0.5 * (gamma_minus_nu_array_ABC[j] - complex_wave_nu_BCE[i][j-1])
        else:
            complex_wave_nu_BCE[i][j] = 0.5*(complex_wave_nu_BCE[i][j-1] + complex_wave_nu_BCE[i-1][j]) + 0.5 * (complex_wave_phi_BCE[i-1][j] - complex_wave_phi_BCE[i][j-1])
            complex_wave_phi_BCE[i][j] = 0.5*(complex_wave_phi_BCE[i][j-1] + complex_wave_phi_BCE[i-1][j]) + 0.5 * (complex_wave_nu_BCE[i-1][j] - complex_wave_nu_BCE[i][j-1])
        complex_wave_mu_BCE[i][j], complex_wave_mach_BCE[i][j] = find_mu_from_nu(complex_wave_nu_BCE[i][j], mach_exit, mach_B+ 10, error_tolerance)
        complex_wave_alpha_BCE[i][j] = calc_positive_slope(complex_wave_mu_BCE[i][j], complex_wave_phi_BCE[i][j])

for j in sp.arange(0, n_characteristics, 1): #iterate over incoming lines
    for i in sp.arange(0, j+1, 1): #iterate over crossings
        if i == j:
            complex_wave_nu_DFG[i][j] =  calc_Prandtl_Meyer_from_Mach(mach_B)
            if i == 0:
                complex_wave_phi_DFG[i][j] = complex_wave_phi_BCE[j][-1] - complex_wave_nu_BCE[j][-1] + complex_wave_nu_DFG[i][j]
            else:
                complex_wave_phi_DFG[i][j] = complex_wave_phi_DFG[i-1][j]  - complex_wave_nu_DFG[i-1][j] + complex_wave_nu_DFG[i][j]
        elif i == 0 :
            complex_wave_nu_DFG[i][j] = 0.5 * (complex_wave_nu_BCE[j][-1] + complex_wave_nu_DFG[i][j-1]) + 0.5 * (complex_wave_phi_DFG[i][j-1] - complex_wave_phi_BCE[j][-1])
            complex_wave_phi_DFG[i][j] = 0.5 * (complex_wave_phi_BCE[j][-1] + complex_wave_phi_DFG[i][j-1]) + 0.5 * (complex_wave_nu_DFG[i][j-1] - complex_wave_nu_BCE[j][-1])
        else:
            complex_wave_nu_DFG[i][j] = 0.5 * (complex_wave_nu_DFG[i][j-1] + complex_wave_nu_DFG[i-1][j]) + 0.5 * (complex_wave_phi_DFG[i][j-1] - complex_wave_phi_DFG[i-1][j])
            complex_wave_phi_DFG[i][j] = 0.5 * (complex_wave_phi_DFG[i][j-1] + complex_wave_phi_DFG[i-1][j]) + 0.5 * (complex_wave_nu_DFG[i][j-1] - complex_wave_nu_DFG[i-1][j])
        complex_wave_mu_DFG[i][j], complex_wave_mach_DFG[i][j] = find_mu_from_nu(complex_wave_nu_DFG[i][j], mach_exit, mach_B+ 10, error_tolerance)
        complex_wave_alpha_DFG[i][j] = calc_negative_slope(complex_wave_mu_DFG[i][j], complex_wave_phi_DFG[i][j])

for j in sp.arange(0, n_characteristics, 1): #iterate over incoming characteristics
    for i in sp.arange(0, j + 1, 1): #iterate over crossings
        if i == j:
            complex_wave_nu_HIJ[i][j] = complex_wave_nu_DFG[j][-1] + complex_wave_phi_DFG[j][-1]
        elif i == 0:
            complex_wave_nu_HIJ[i][j] = 0.5*(complex_wave_nu_HIJ[i][j-1] + complex_wave_nu_DFG[j][-1]) + 0.5 * (complex_wave_phi_DFG[j][-1] - complex_wave_phi_HIJ[i][j-1])
            complex_wave_phi_HIJ[i][j] = 0.5*(complex_wave_phi_HIJ[i][j-1] + complex_wave_phi_DFG[j][-1]) + 0.5 * (complex_wave_nu_DFG[j][-1] - complex_wave_nu_HIJ[i][j-1])
        else:
            complex_wave_nu_HIJ[i][j] = 0.5*(complex_wave_nu_HIJ[i][j-1] + complex_wave_nu_HIJ[i-1][j]) + 0.5 * (complex_wave_phi_HIJ[i-1][j] - complex_wave_phi_HIJ[i][j-1])
            complex_wave_phi_HIJ[i][j] = 0.5*(complex_wave_phi_HIJ[i][j-1] + complex_wave_phi_HIJ[i-1][j]) + 0.5 * (complex_wave_nu_HIJ[i-1][j] - complex_wave_nu_HIJ[i][j-1])
        complex_wave_mu_HIJ[i][j], complex_wave_mach_HIJ[i][j] = find_mu_from_nu(complex_wave_nu_HIJ[i][j], mach_exit, mach_B+ 10, error_tolerance)
        complex_wave_alpha_HIJ[i][j] = calc_positive_slope(complex_wave_mu_HIJ[i][j], complex_wave_phi_HIJ[i][j])

# =============================================================================
# Finding characteristic intersections BCE
# =============================================================================
gammas = sp.zeros([n_characteristics,len(x)])
jet_boundary = height + sp.tan(degrees_to_radians(phi_B)) * x #function for initial jet_boundary
for i in sp.arange(0, n_characteristics, 1): #iterate over incoming characteristics
    for j in sp.arange(i, n_characteristics, 1):
        positive_slope = 0
        negative_slope = 0
        if i == 0:
            negative_slope = calc_negative_slope(calc_mu_from_Mach(gamma_minus_mach_array_ABC[j]), gamma_minus_phi_array_ABC[j])
            incoming_characteristic = height + negative_slope * x
            
            if j == 0:
                reflected_characteristic = 0 * x
            else: 
                prev_x = x_intersections_BCE[i][j-1]
                positive_slope = calc_positive_slope(complex_wave_mu_BCE[i][j], complex_wave_phi_BCE[i][j])
                diff_slopes_positive = y_intersections_BCE[i][j-1] - positive_slope * x_intersections_BCE[i][j-1]
                reflected_characteristic = positive_slope * x + diff_slopes_positive 
        else:
            negative_slope = calc_negative_slope(complex_wave_mu_BCE[i][j], complex_wave_phi_BCE[i][j])
            if j == i:
                positive_slope = 0
            else:
                positive_slope = calc_positive_slope(complex_wave_mu_BCE[i][j], complex_wave_phi_BCE[i][j])

            incoming_characteristic = y_intersections_BCE[i-1][j] + negative_slope * (x - x_intersections_BCE[i-1][j])
            reflected_characteristic = y_intersections_BCE[i][j-1] + positive_slope * (x - x_intersections_BCE[i][j-1])
        y_idx = sp.argwhere(sp.diff(sp.sign(incoming_characteristic - reflected_characteristic))).flatten()
        y_intersections_BCE[i][j] = reflected_characteristic[y_idx]
        
        if i > 0:
            x_intersections_BCE[i][j] = x[y_idx] 
        else: 
            x_intersections_BCE[i][j] = x[y_idx]
            

# =============================================================================
#       Finding the arrays to plot the n_characteristics of BCE
# =============================================================================
        x_idx = sp.where(x >= x_intersections_BCE[i][j])
        if i ==0 : #first wave intersection
            gammas[j][:y_idx[0]]= height + negative_slope * x[:y_idx[0]] 
            prev_x_idx_reflected = sp.where(x >= x_intersections_BCE[i][j-1])
            if j > 0:
                gammas[i][prev_x_idx_reflected[0][0]:] = y_intersections_BCE[i][j-1] + positive_slope * (x[prev_x_idx_reflected[0][0]:] - x[prev_x_idx_reflected[0][0]])

        elif (j == i): #intersection with x-axis
            prev_x_idx = sp.where(x >= x_intersections_BCE[i-1][j])
            gammas[j][prev_x_idx[0][0]:] = y_intersections_BCE[i-1][j] + negative_slope * (x[prev_x_idx[0][0]:] - x[prev_x_idx[0][0]])

        else:
            prev_x_idx = sp.where(x >= x_intersections_BCE[i-1][j])
            prev_x_idx_reflected = sp.where(x >= x_intersections_BCE[i][j-1])
            idx = sp.where(x >= x_intersections_BCE[i][j])
            gammas[j][prev_x_idx[0][0]:]= y_intersections_BCE[i-1][j] + negative_slope * (x[prev_x_idx[0][0]:] - x[prev_x_idx[0][0]])
            gammas[i][prev_x_idx_reflected[0][0]:]= y_intersections_BCE[i][j-1] + positive_slope * (x[prev_x_idx_reflected[0][0]:] - x[prev_x_idx_reflected[0][0]])


# =============================================================================
# Finding characteristic intersections DFG
# =============================================================================
for i in sp.arange(0, n_characteristics, 1):
    for j in sp.arange(i, n_characteristics, 1):
        diff_slopes_positive = 0 
        diff_slopes_negative = 0
        if i == 0:
            diff_slopes = y_intersections_DFG[i][j-1] - complex_wave_alpha_BCE[j][-1] * x_intersections_DFG[i][j-1]
            index_x_prev_wave = sp.where(x >= x_intersections_DFG[i][j-1])
            incoming_characteristic = make_linear_function(complex_wave_alpha_BCE[j][-1], y_intersections_BCE[j][-1], x, x_intersections_BCE[j][-1])
            positive_slope = calc_positive_slope(complex_wave_mu_DFG[i][j], complex_wave_phi_DFG[i][j])
            negative_slope = calc_negative_slope(complex_wave_mu_DFG[i][j], complex_wave_phi_DFG[i][j])
            if j == i:
                reflected_characteristic = jet_boundary
            else: 
                prev_x = x_intersections_DFG[i][j-1]
                negative_slope = calc_negative_slope(complex_wave_mu_DFG[i][j], complex_wave_phi_DFG[i][j])
                diff_slopes_negative = y_intersections_DFG[i][j-1] - negative_slope * x_intersections_DFG[i][j-1]
                reflected_characteristic = make_linear_function(negative_slope, diff_slopes_negative, x, 0)
        else:
            negative_slope = calc_negative_slope(complex_wave_mu_DFG[i][j], complex_wave_phi_DFG[i][j])
            positive_slope = calc_positive_slope(complex_wave_mu_DFG[i][j], complex_wave_phi_DFG[i][j])                    
            if j != i:               
                diff_slopes_positive = y_intersections_DFG[i-1][j] - positive_slope * x_intersections_DFG[i-1][j]
                diff_slopes_negative = y_intersections_DFG[i][j-1] - negative_slope * x_intersections_DFG[i][j-1]
                incoming_characteristic = make_linear_function(positive_slope, diff_slopes_positive, x,  0)
                reflected_characteristic = make_linear_function(negative_slope, diff_slopes_negative, x, 0)
            else: 
                reflected_characteristic= jet_boundary
                diff_slopes_positive = y_intersections_DFG[i-1][j] - positive_slope * x_intersections_DFG[i-1][j]
                incoming_characteristic = make_linear_function(positive_slope, diff_slopes_positive, x,  0)
                
        y_idx = sp.argwhere(sp.diff(sp.sign(reflected_characteristic - incoming_characteristic))).flatten()
        y_intersections_DFG[i][j] = incoming_characteristic[y_idx]

        if i > 0:
            x_intersections_DFG[i][j] = x[y_idx]
        else: 
            x_intersections_DFG[i][j] = x[y_idx]

# =============================================================================
#       Finding the arrays to plot the n_characteristics DFG
# =============================================================================

        x_idx = sp.where(x >= x_intersections_DFG[i][j])
        if i ==0 : #first wave intersection
            prev_wave_x_idx = sp.where(x >= x_intersections_BCE[j][-1])
            prev_x_idx_reflected = sp.where(x >= x_intersections_DFG[i][j-1])
            gammas[j][prev_wave_x_idx[0][0]:x_idx[0][0]]= y_intersections_BCE[j][-1] + positive_slope * (x[prev_wave_x_idx[0][0]:x_idx[0][0]] - x[prev_wave_x_idx[0][0]])
            if j > 0:
                gammas[j][prev_wave_x_idx[0][0]:x_idx[0][0]]= y_intersections_BCE[j][-1]  + complex_wave_alpha_BCE[j][-1]* (x[prev_wave_x_idx[0][0]:x_idx[0][0]] - x[prev_wave_x_idx[0][0]])
                gammas[i][prev_x_idx_reflected[0][0]:x_idx[0][0] + 1]= y_intersections_DFG[i][j-1] + negative_slope * (x[prev_x_idx_reflected[0][0]:x_idx[0][0] + 1] - x[prev_x_idx_reflected[0][0]])
            else:
                diff_slopes = jet_boundary[x_idx[0][0]] - x[x_idx[0][0]] * sp.tan(degrees_to_radians(complex_wave_phi_DFG[i][j]))
                jet_boundary[x_idx[0][0]:] = sp.tan(degrees_to_radians(complex_wave_phi_DFG[i][j])) * x[x_idx[0][0]:] + diff_slopes

        elif (j == i): #intersection with jet-boundary
            prev_x_idx = sp.where(x >= x_intersections_DFG[i-1][j])
            gammas[j][prev_x_idx[0][0]:x_idx[0][0] +1]= y_intersections_DFG[i-1][j] + positive_slope * (x[prev_x_idx[0][0]:x_idx[0][0] +1] - x[prev_x_idx[0][0]])
            diff_slopes = jet_boundary[x_idx[0][0]] - x[x_idx[0][0]] * sp.tan(degrees_to_radians(complex_wave_phi_DFG[i][j]))
            jet_boundary[x_idx[0][0]:] = sp.tan(degrees_to_radians(complex_wave_phi_DFG[i][j])) * x[x_idx[0][0]:] + diff_slopes

        else:
            prev_x_idx = sp.where(x >= x_intersections_DFG[i-1][j])
            idx = sp.where(x >= x_intersections_DFG[i][j])
            prev_x_idx_reflected = sp.where(x >= x_intersections_DFG[i][j-1])
            gammas[i][prev_x_idx_reflected[0][0]:]= negative_slope * (x[prev_x_idx_reflected[0][0]:] ) + diff_slopes_negative 
            gammas[j][prev_x_idx[0][0]:]= positive_slope * (x[prev_x_idx[0][0]:] ) + diff_slopes_positive 

        if j == n_characteristics - 1:
            gammas[i][x_idx[0][0]:]= y_intersections_DFG[i][j] + complex_wave_alpha_DFG[i][j] * (x[x_idx[0][0]:]- x[x_idx[0][0]])


# =============================================================================
# Finding the characteristic intersections in HIJ
# =============================================================================
for i in sp.arange(0, n_characteristics, 1):
    for j in sp.arange(i, n_characteristics, 1):
        
        diff_slopes_positive = 0 
        diff_slopes_negative = 0
        positive_slope = 0
        negative_slope = 0
        if i == 0:
            negative_slope = complex_wave_alpha_DFG[j][-1]
            diff_slopes_negative = y_intersections_DFG[j][-1] - negative_slope * x_intersections_DFG[j][-1]
            incoming_characteristic = diff_slopes_negative + negative_slope * x
            
            if j == 0:
                reflected_characteristic = 0 * x
            else: 
                positive_slope = calc_positive_slope(complex_wave_mu_HIJ[i][j-1], complex_wave_phi_HIJ[i][j-1])
                diff_slopes_positive = y_intersections_HIJ[i][j-1] - positive_slope * x_intersections_HIJ[i][j-1]
                reflected_characteristic = positive_slope * x + diff_slopes_positive

        else:
            negative_slope = calc_negative_slope(complex_wave_mu_HIJ[i-1][j], complex_wave_phi_HIJ[i-1][j])
            if j == i:
                reflected_characteristic = 0 * x
            else:
                positive_slope = calc_positive_slope(complex_wave_mu_HIJ[i][j], complex_wave_phi_HIJ[i][j])
                diff_slopes_positive = y_intersections_HIJ[i][j-1] - positive_slope * x_intersections_HIJ[i][j-1]
                reflected_characteristic = y_intersections_HIJ[i][j-1] + positive_slope * (x - x_intersections_HIJ[i][j-1])

            incoming_characteristic = y_intersections_HIJ[i-1][j] + negative_slope * (x - x_intersections_HIJ[i-1][j])
            
        y_idx = sp.argwhere(sp.diff(sp.sign(incoming_characteristic - reflected_characteristic))).flatten()
        y_intersections_HIJ[i][j] = reflected_characteristic[y_idx]
        
        if i > 0:
            x_intersections_HIJ[i][j] = x[y_idx] 
        else: 
            x_intersections_HIJ[i][j] = x[y_idx]


# =============================================================================
#       Finding the arrays to plot the n_characteristics of HIJ
# =============================================================================
        x_idx = sp.where(x >= x_intersections_HIJ[i][j])
        
        if i ==0 : #first wave intersection
            prev_wave_x_idx = sp.where(x >= x_intersections_DFG[j][-1])
            gammas[j][prev_wave_x_idx[0][0]:]= diff_slopes_negative + negative_slope * x[prev_wave_x_idx[0][0]:] 

            if j > 0:
                prev_x_idx_reflected = sp.where(x >= x_intersections_HIJ[i][j-1])
                gammas[i][prev_x_idx_reflected[0][0]:] = y_intersections_HIJ[i][j-1] + positive_slope * (x[prev_x_idx_reflected[0][0]:] - x[prev_x_idx_reflected[0][0]])

        elif (j == i): #intersection with x-axis
            prev_x_idx = sp.where(x >= x_intersections_HIJ[i-1][j])
            gammas[j][prev_x_idx[0][0]:]= y_intersections_HIJ[i-1][j] + negative_slope * (x[prev_x_idx[0][0]:] - x[prev_x_idx[0][0]])
            #gammas[j][prev_x_idx[0][0]:] = y_intersections_HIJ[i-1][j] + negative_slope * (x[prev_x_idx[0][0]:] - x[prev_x_idx[0][0]])

        else:
            prev_x_idx = sp.where(x >= x_intersections_HIJ[i-1][j])
            prev_x_idx_reflected = sp.where(x >= x_intersections_HIJ[i][j-1])
            idx = sp.where(x >= x_intersections_HIJ[i][j])
            gammas[j][prev_x_idx[0][0]:]= y_intersections_HIJ[i-1][j] + negative_slope * (x[prev_x_idx[0][0]:] - x[prev_x_idx[0][0]])
            gammas[i][prev_x_idx_reflected[0][0]:]= y_intersections_HIJ[i][j-1] + positive_slope * (x[prev_x_idx_reflected[0][0]:] - x[prev_x_idx_reflected[0][0]])
        if j == n_characteristics - 1:
            gammas[i][x_idx[0][0]:]= y_intersections_HIJ[i][j] + complex_wave_alpha_HIJ[i][j] * (x[x_idx[0][0]:]- x[x_idx[0][0]])

# =============================================================================
#       Finding the streamline properties
# =============================================================================
def find_streamline(starting_height):
    streamline = starting_height + 0 * x
    press_dist = sp.array((pressure_of_total_pressure(mach_exit)))
    x_crossings = sp.array(x[0])
    
    
    if starting_height > 0:
        x_end_not_reached = True 
    else:
        x_end_not_reached = False
        press_dist, x_crossings = pressure_distribution_centre(press_dist, x_crossings)
    prev_idx = 0

    def find_streamline_crossing(streamline, prev_idx):
        streamline_indices = sp.ndarray([0])
        for i in sp.arange(0, n_characteristics, 1): 
            crossing_index = sp.argwhere(sp.diff(sp.sign(streamline[prev_idx:] - gammas[i][prev_idx:]))).flatten()
            if len(streamline_indices) == 0 and crossing_index.size > 0:
                streamline_indices = [{
                    "index": prev_idx + min(crossing_index), 
                    "gamma": i
                }]
            elif crossing_index.size > 0:
                streamline_indices = sp.append(streamline_indices, {
                    "index": prev_idx + min(crossing_index), 
                    "gamma": i
                    })
        return streamline_indices

    def find_crossing(streamline_indices, prev_idx):
        lowest_idx = steps_x
        gamma = 0
        for i in sp.arange(0, len(streamline_indices), 1):
            if streamline_indices[i]["index"] > prev_idx and streamline_indices[i]["index"] < lowest_idx:
                lowest_idx = streamline_indices[i]["index"]
                gamma = streamline_indices[i]["gamma"]
        return lowest_idx, gamma

    slope = 0
    while x_end_not_reached:
        streamline_indices = find_streamline_crossing(streamline, prev_idx)
        prev_idx, gamma = find_crossing(streamline_indices, prev_idx)
        if x[prev_idx] < x_intersections_BCE[gamma][gamma]:
            slope = sp.tan(degrees_to_radians(gamma_minus_phi_array_ABC[gamma]))
            current_mach = gamma_minus_mach_array_ABC[gamma]
        elif x[prev_idx] > x_intersections_BCE[gamma][gamma] and x[prev_idx] < x_intersections_BCE[gamma][-1]:
            for i in sp.arange(0, n_characteristics, 1):
                if streamline[prev_idx] < y_intersections_BCE[gamma][i] and streamline[prev_idx] > y_intersections_BCE[gamma][i-1]:
                    slope = sp.tan(degrees_to_radians((complex_wave_phi_BCE[gamma][i] + complex_wave_phi_BCE[gamma][i-1])/2))
                    current_mach = complex_wave_mach_BCE[gamma][i]
        elif x[prev_idx] > x_intersections_BCE[gamma][-1] and x[prev_idx] < x_intersections_DFG[gamma][gamma]:
            slope = sp.tan(degrees_to_radians(complex_wave_phi_BCE[gamma][-1]))
            current_mach = complex_wave_mach_BCE[gamma][-1]
        elif x[prev_idx] > x_intersections_DFG[gamma][gamma] and x[prev_idx] < x_intersections_DFG[gamma][-1]:
            for i in sp.arange(0, n_characteristics, 1):
                if (
                        streamline[prev_idx] < y_intersections_DFG[gamma][i] and streamline[prev_idx] > y_intersections_DFG[gamma][i-1]
                        or streamline[prev_idx] > y_intersections_DFG[gamma][i] and streamline[prev_idx] < y_intersections_DFG[gamma][i-1]
                    ):
                    slope = sp.tan(degrees_to_radians(complex_wave_phi_DFG[gamma][i] + complex_wave_phi_DFG[gamma][i-1])/2)
                    current_mach = complex_wave_mach_DFG[gamma][i]
        elif x[prev_idx] > x_intersections_DFG[gamma][-1] and x[prev_idx] < x_intersections_HIJ[gamma][gamma]:
            slope = sp.tan(degrees_to_radians(complex_wave_phi_DFG[gamma][-1]))
            current_mach = complex_wave_mach_DFG[gamma][-1]
        elif x[prev_idx] > x_intersections_HIJ[gamma][gamma] and x[prev_idx] < x_intersections_HIJ[gamma][-1]:
            for i in sp.arange(0, n_characteristics, 1):
                if streamline[prev_idx] < y_intersections_HIJ[gamma][i] and streamline[prev_idx] > y_intersections_HIJ[gamma][i-1]:
                    slope = sp.tan(degrees_to_radians(complex_wave_phi_HIJ[gamma][i] + complex_wave_phi_HIJ[gamma][i-1])/2)
                    current_mach = complex_wave_mach_HIJ[gamma][i]
        elif x[prev_idx] > x_intersections_HIJ[gamma][-1]:
            slope = sp.tan(degrees_to_radians(complex_wave_phi_HIJ[gamma][-1]))
            x_end_not_reached = False

        streamline[prev_idx:] = streamline[prev_idx] + slope * (x[prev_idx:] - x[prev_idx])
        
        x_crossings = sp.hstack((x_crossings, x[prev_idx]))
        press_dist = sp.hstack((press_dist, pressure_of_total_pressure(current_mach)))
        
    return [streamline, press_dist, x_crossings]


# =============================================================================
#           Plotting the n_characteristics over the entire domain
# =============================================================================
debugVar = sp.ndarray([1, n_characteristics])
def create_plot(streamlines= []):
    fig,ax = plt.subplots(1)
    plt.ylabel('y-direction')
    plt.xlabel('x-direction')
    plt.title('Jet flow downstream')
    for i in sp.arange(0, n_characteristics, 1):
        if i == 0 or i == n_characteristics -1:
            plt.plot(x, gammas[i], '-', color="blue")
        else:
            plt.plot(x, gammas[i], '--', color="blue")
    plt.plot(x, jet_boundary, '--', color="red")


    pc = ax.pcolormesh(x_intersections_HIJ.astype('float'), y_intersections_HIJ.astype('float'), complex_wave_mach_HIJ.astype('float'), vmin=mach_exit) 
    print(sp.rot90(sp.fliplr(x_intersections_HIJ)))
    fig.colorbar(pc)
    ax.pcolormesh(x_intersections_BCE.astype('float'), y_intersections_BCE.astype('float'), complex_wave_mach_BCE.astype('float'), vmin=mach_exit) 
    print(sp.rot90(sp.fliplr(x_intersections_DFG.astype('float'))), sp.rot90(sp.fliplr(y_intersections_DFG.astype('float'))), sp.rot90(sp.fliplr(complex_wave_mach_DFG.astype('float'))))
    ax.pcolormesh(sp.rot90(sp.fliplr(x_intersections_DFG.astype('float'))), sp.rot90(sp.fliplr(y_intersections_DFG.astype('float'))), sp.rot90(sp.fliplr(complex_wave_mach_DFG.astype('float'))), vmin=mach_exit)

    if(len(streamlines) > 0):
        for i in sp.arange(0, len(streamlines), 1):
            plt.plot(x, streamlines[i][0], color="green")
        plt.show()
        fig,ax = plt.subplots(1)
        plt.ylabel('P/P_T')
        plt.xlabel('x-direction')
        plt.title('Local pressure')
        plt.ylim(bottom=0, top=0.04)
        for i in sp.arange(0, len(streamlines), 1):
            plt.plot(streamlines[i][2], streamlines[i][1])
        plt.legend(['Centre-line', 'H/2'])
    plt.show()
    
create_plot()
#create_plot([find_streamline(0), find_streamline(height/2)])
