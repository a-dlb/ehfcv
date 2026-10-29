#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:42:10 2026

@author: antoinedlbck
"""

from vehicle import Vehicle
from motor import Motor
from constants import GEAR_PAIR_EFF

# Simple helper to make the prints clearer :) 
def print_section(title):
    
    print("\n" + "=" * 30)
    print(f"{title.upper()}") #.upper() = change the string into uppercase
    print("=" * 30)
    
    return 
    

def main():
    
    car = Vehicle()
    motor = Motor()

    # Rotor inertia
    J_m_full = motor.rotor_inertia_cylinder
    J_m_hollow = motor.rotor_inertia_hollow
    J_m_NX = motor.rotor_inertia
    
    print_section("Rotor inertia")
    
    # - Full cylinder inertia
    print("Full cylinder inertia:")
    print(f"  J_(m, homogeneous) = {J_m_full:.5f} (kg.m^2) \n")
    
    # - Hollow cylinder inertia
    print("Hollow cylinder inertia:")
    print(f"  J_(m, hollow) = {J_m_hollow:.5f} (kg.m^2) \n")
    
    # - Inertia from NX model
    print("Inertia from NX model:")
    print(f"  J_(m, CAD) = {J_m_NX:.5f} (kg.m^2) \n")
    
    # - Hollow cylinder inertia
    
    # - NX model inertia
    
    
    # Effective rolling radius
    F_zf = car.vertical_force_front
    F_zr = car.vertical_force_rear

    R_u = car.tire_unloaded_radius

    R_ef = car.tire_effective_radius_front
    R_er = car.tire_effective_radius_rear
    
    print_section("Effective rolling radius")
    
    # - Vertical forces
    print("Vertical force on a tire : \n")
    print(f"  front: F_zf = {F_zf:.1f} (N)")
    print(f"  rear:  F_zr = {F_zr:.1f} (N)")   
    print("\n")
    
    # - Unloaded rolling radius
    print("Unloaded rolling radius : \n")
    print(f"  R_u = {R_u * 1e3:.1f} (mm)")
    print("\n")
    
    # - Effective rolling radii
    print("Effective rolling radii : \n")
    print(f"  front: R_ef = {R_ef * 1e3:.1f} (mm)")
    print(f"  rear:  R_er = {R_er * 1e3:.1f} (mm)")
    
    
    # Transmission efficiency
    eta_tr = car.transmission_eff
    
    print_section("Transmission efficiency")
    print("Gear pair efficiency:")
    print(f"  eta_gears = eta_diff = {GEAR_PAIR_EFF} \n")
    print("Transmission efficiency:")
    print(f"  eta_tr = {eta_tr * 1e2 :.2f} %")
    
    return


if __name__ == "__main__":
    
    main()