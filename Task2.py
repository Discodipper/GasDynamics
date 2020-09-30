# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 20:40:01 2020

@author: Thomas Lokken 4449282
"""

import scipy as sp

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

# calculate the Prandtl-meyer angle
def calc_Mach_from_Prandtl_Meyer(mu):
    labda = sp.sqrt((gamma+1)/(gamma-1))
    mu_rad = degrees_to_radians(mu)
    return sp.tan(mu_rad/labda) / (1/labda - 1)

def calc_alpha(mach, phi):
    return calc_mu_from_Mach(mach) - phi

def calc_forward_phi_gamma_plus(nu_backward, phi_backward, nu_forward):
    return - (nu_backward - phi_backward - nu_forward)

def bisection_step_function(mach, alpha, backward_phi, backward_nu):
    return calc_Prandtl_Meyer_from_Mach(mach) - calc_mu_from_Mach(mach) + alpha + backward_phi - backward_nu

def bisection_method(mach_head, mach_tail, tolerance, alpha, backward_phi, backward_nu):
    print(mach_head, mach_tail, tolerance, alpha, backward_phi, backward_nu)
    mach_left = mach_head
    mach_right = mach_tail
    while (sp.absolute(mach_left - mach_right) >= tolerance):
        output_mach = (mach_left + mach_right)/2.0
        step = bisection_step_function(output_mach, alpha, backward_phi, backward_nu)
        print(bisection_step_function(mach_left, alpha, backward_phi, backward_nu))
        if step == 0: 
            break
        if step * bisection_step_function(mach_left, alpha, backward_phi, backward_nu) < 0:
            mach_right = output_mach
        else:
            mach_left = output_mach
    print(output_mach)
    mu = calc_mu_from_Mach(output_mach)
    phi = mu - alpha
    return output_mach, phi, mu
        
        
    

# =============================================================================
# Iteration parameters
# =============================================================================
characteristics = 10
mach_array = sp.zeros([characteristics,2])
phi_array = sp.zeros([characteristics,2])
mu_array = sp.zeros([characteristics,2])
mach_array[0] = mach_exit
phi_array[0] = phi_exit
mu_array[0] = calc_mu_from_Mach(mach_exit)
error_tolerance = 0.001

# =============================================================================
# Necessary variables to perform iteration
# =============================================================================
mach_B = sp.sqrt((1-2**(-(gamma-1)/gamma))/((gamma-1)/2) + mach_exit**2)
phi_B = calc_forward_phi_gamma_plus(calc_Prandtl_Meyer_from_Mach(mach_exit), phi_exit, calc_Prandtl_Meyer_from_Mach(mach_B))
delta_alpha = calc_alpha(mach_B, phi_B) - calc_alpha(mach_exit, phi_exit)
alpha_0 =  calc_alpha(mach_exit, phi_exit)
print(sp.arange(
        alpha_0,
        alpha_0 + delta_alpha + delta_alpha / (characteristics - 1), 
        delta_alpha / characteristics
    ))

for j, alpha in enumerate(sp.arange(
        alpha_0,
        alpha_0 + delta_alpha + delta_alpha / (characteristics - 1),
        delta_alpha / characteristics
    )):
# =============================================================================
#     print (mach_array[j - 1, 0] if j > 0 else mach_array[j, 0])
# =============================================================================
    nu = calc_Prandtl_Meyer_from_Mach(mach_array[j - 1, 0] if j > 0 else mach_exit)
    print(nu)
    mach_array[j, 0], phi_array[j, 0] = bisection_method(
        mach_array[j - 1, 0] if j > 0 else mach_exit, 
        mach_B, 
        error_tolerance, 
        alpha, 
        phi_array[j - 1, 0] if j > 0 else phi_exit, 
        nu    
    )
    for i in sp.arange(0, j + 1, 1):
        x = 2
        
        

print(mach_array)
    


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

