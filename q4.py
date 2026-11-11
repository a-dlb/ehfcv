#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:35:28 2026

@author: antoinedlbck
"""

import numpy as np
import matplotlib.pyplot as plt
from vehicle import Vehicle
from motor import Motor
from plots import plot_netforce, plot_acc, plot_acc_capa
from plot_style import get_style
from resistive_forces import grading_force, net_force
from constants import KMH_TO_MS, RADS_TO_RPM

PLOT = True
EXPORT = True

def main():
    
    vehicle = Vehicle()
    motor = Motor()
    
    slopes = [0, 5, 10] # %
    speed = np.linspace(1e-6, vehicle.max_speed, 1000) # km/h
    speed_ms = speed * KMH_TO_MS # m/s
    
    V_max_kmh = vehicle.max_speed
    
    rpm_base = motor.base_speed_rpm
    R_e = vehicle.tire_effective_radius_rear
    i_red = vehicle.gear_ratio
    m_c = vehicle.vehicle_mass
    m_et = vehicle.total_effective_mass(motor)
    m_e = m_et - m_c
    gamma = m_e/m_c
    
    V_base_kmh = rpm_base * 2*np.pi/60 * R_e / i_red * 3.6
    
    rpm = ((speed * KMH_TO_MS) * i_red / R_e) * RADS_TO_RPM 
    
    F_tract = vehicle.tractive_force(motor, rpm)
    
    print(f"\n===== Valeurs Q3 (V_max = {V_max_kmh} km/h) =====")
    print(f"\ngamma = {gamma:.4f}, m_e = {m_e:.1f} kg, m_e,t = {m_et:.1f} kg")
    print(f"V_base                 : {V_base_kmh:.1f} km/h")
    print(f"F_tract max (V < V_base): {F_tract[0]:.0f} N")
    print(f"F_tract at V_max       : {F_tract[-1]:.0f} N")
    
    for s in slopes:
        
        F_net   = net_force(vehicle, motor, speed, s)
        a_x     = F_net / m_et          # (m/s^2)
        inv_a   = 1 / a_x               # (s^2/m)
        F_grade = grading_force(vehicle, s) # returns a scalar
        F_grade = np.full_like(speed, F_grade)
    
        t_base = np.trapezoid(inv_a[speed <= V_base_kmh], speed_ms[speed <= V_base_kmh])
        t_100 = np.trapezoid(inv_a[speed <= 100], speed_ms[speed <= 100])
        t_max = np.trapezoid(inv_a, speed_ms)
    
        print(f"\n--- Slope {s*100:.0f} % ---")
        print(f"Grading resistance : {F_grade[0]:.1f} N")
        print(f"F_net at V = 0     : {F_net[0]:.1f} N")
        print(f"F_net at V_base    : {np.interp(V_base_kmh, speed, F_net):.1f} N")
        print(f"F_net at V_max     : {F_net[-1]:.1f} N")
        print(f"a_x at V = 0       : {a_x[0]:.2f} m/s^2")
        print(f"a_x at V_max       : {a_x[-1]:.2f} m/s^2")
        print(f"1/a_x at V = 0     : {inv_a[0]:.3f} s^2/m")
        print(f"1/a_x at V_max     : {inv_a[-1]:.3f} s^2/m")
        print(f"t(0-V_base)        : {t_base:.1f} s")
        print(f"t(0-100 km/h)      : {t_100:.1f} s")
        print(f"t(0-{V_max_kmh} km/h)      : {t_max:.1f} s")
    
    # Max speed limited by motor redline
    rpm_max = motor.max_rpm
    V_max_motor_kmh = rpm_max * 2*np.pi/60 * R_e / i_red * 3.6
    print(f"\nMotor speed at V_max   : {rpm[-1]:.0f} rpm (rpm_max = {rpm_max} rpm)")
    print(f"Speed at rpm_max       : {V_max_motor_kmh:.1f} km/h")
    
    
    if PLOT:
        
        with plt.rc_context(get_style()):
        
            regimes = ["peak", "nom"]
        
            for regime in regimes:
                
                path = f"Graphs/Net_forceQ4_{regime}.pdf" if EXPORT else None
                plot_netforce(vehicle, motor, speed, slopes, highlight=regime, export_path=path)
                
                path = f"Graphs/acceleration_{regime}.pdf" if EXPORT else None
                plot_acc(vehicle, motor, speed, slopes, highlight=regime, export_path=path)
                
                path = f"Graphs/Inverse_accelerationQ4_{regime}.pdf" if EXPORT else None
                plot_acc_capa(vehicle, motor, speed, slopes, highlight=regime, export_path=path)

    return


if __name__ == "__main__":
    
    main()