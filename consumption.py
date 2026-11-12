#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 20:50:42 2026

@author: claravliegen
"""

import numpy as np
import pandas as pd
from resistive_forces import total_force
from constants import (KMH_TO_MS, RADS_TO_RPM, P_ACCESSORIES, 
                       ALPHA_REGEN, P_REGEN_MAX)

#-------------------------------- calcul (a) -------------------------------------
def motor_efficiency(motor, rpm, torque):
    """
    Motor efficiency eta_m(N, T) (-), from the order 2 polynomial 
    approximating the efficiency map (Appendix B)
    
      eta_m = C0 + C1 N + C2 T + C3 N^2 + C4 N T + C5 T^2
      
    with N the motor speed in (rpm) and T the motor torque in (Nm)
    """
    
    C0, C1, C2, C3, C4, C5 = motor.eff_coeffs
    N = rpm
    T = torque
    
    return C0 + C1 * N + C2 * T + C3 * N ** 2 + C4 * N * T + C5 * T ** 2

def motor_operating_point(vehicle, speed, slope=0):
    """
    Motor operating point at constant speed: 
    rotation speed (rpm) and torque C_m (Nm)
    (!) speed is expected in (km/h) (and slope in (%))
    
      omega_m = V i / R_e
      C_m     = R_e F_t / (eta_t i)
    """
    
    i = vehicle.gear_ratio
    R_er = vehicle.tire_effective_radius_rear
    eta_tr = vehicle.transmission_eff
    
    # At constant speed, tractive force = sum of the resistive forces
    F_t = total_force(vehicle, speed, slope)
    
    rpm = (speed * KMH_TO_MS * i / R_er) * RADS_TO_RPM
    torque = R_er * F_t / (eta_tr * i)
    
    return rpm, torque

def battery_power_constant_speed(vehicle, motor, speed, slope=0,  P_acc=P_ACCESSORIES):
    """
    Power drawn from the battery P_bat (W) at constant speed
    (!) speed is expected in (km/h) (and slope in (%))
    
      P_bat = F_t V / (eta_t eta_m(C_m, omega_m)) + P_accessories
    """
    
    eta_tr = vehicle.transmission_eff
    speed_ms = speed * KMH_TO_MS
    
    F_t = total_force(vehicle, speed, slope)
    
    rpm, torque = motor_operating_point(vehicle, speed, slope)
    eta_m = motor_efficiency(motor, rpm, torque)
    
    return F_t * speed_ms / (eta_tr * eta_m) + P_acc


def consumption_constant_speed(vehicle, motor, speed, slope=0, P_acc=P_ACCESSORIES ):
    """
    Energy consumption E (kWh/km) at constant speed
    (!) speed is expected in (km/h) (and slope in (%))
    
      E = P_bat (kW) / V (km/h)
    """
    
    P_bat = battery_power_constant_speed(vehicle, motor, speed, slope, P_acc)
    
    return (P_bat * 1e-3) / speed





#-------------------------------- cycle WLTP -------------------------------------
def load_wltp(path):
    """
    Reads the Excel file of the WLTP cycle.
    Columns: phase, total elapsed time (s), phase elapsed time (s), speed (km/h)
    (the first two rows are the titles and the units)
    
    Returns t (s), speed (km/h), phase (Low, Medium, High or Extra-high)
    """
    
    df = pd.read_excel(path, header=None, skiprows=2,
                       names=["phase", "t", "t_phase", "speed"])
    
    t = df["t"].to_numpy(dtype=float)
    speed = df["speed"].to_numpy(dtype=float)
    phase = df["phase"].to_numpy()
    
    return t, speed, phase
 
 
def battery_power_cycle(vehicle, motor, t, speed, slope=0, alpha=ALPHA_REGEN):
    """
    Power at the battery P_bat (W) along a driving cycle
    (> 0: drawn from the battery, < 0: sent back to the battery)
    (!) t is expected in (s), speed in (km/h) (and slope in (%))
    
      F_t = F_res + m_et dV/dt
      
      traction (F_t > 0): P_bat = F_t V / (eta_t eta_m)       + P_accessories
      braking  (F_t < 0): P_bat = alpha eta_t eta_m F_t V     + P_accessories
      
    The recovered power is limited to P_REGEN_MAX.
    """
    
    i = vehicle.gear_ratio
    R_er = vehicle.tire_effective_radius_rear
    eta_tr = vehicle.transmission_eff
    m_et = vehicle.total_effective_mass(motor)
    
    speed_ms = speed * KMH_TO_MS
    accel = np.gradient(speed_ms, t) # dV/dt (m/s^2)
    
    # Tractive force required at the wheels (0 when the vehicle is stopped)
    F_t = total_force(vehicle, speed, slope) + m_et * accel
    F_t = np.where(speed > 0, F_t, 0)
    
    # Motor operating point 
    rpm = (speed_ms * i / R_er) * RADS_TO_RPM
    torque = np.abs(R_er * F_t / (eta_tr * i))
    
    # np.maximum avoids a division by 0 in motor.torque when the vehicle is stopped
    if np.any(torque > motor.torque(np.maximum(rpm, 1e-6))):
        raise ValueError("The cycle requires more than the motor peak torque")
        
    eta_m = motor_efficiency(motor, rpm, torque)
    
    P_wheels = F_t * speed_ms
    
    P_traction = P_wheels / (eta_tr * eta_m)
    P_braking = np.maximum(alpha * eta_tr * eta_m * P_wheels, -P_REGEN_MAX)
    
    return np.where(F_t >= 0, P_traction, P_braking) + P_ACCESSORIES
 
 
def cycle_energy(t, speed, P_bat):
    """
    Energies over a driving cycle (or a part of it)
    (!) t is expected in (s), speed in (km/h) and P_bat in (W)
    
    Returns
      distance (km)
      E_out (kWh) : energy drawn from the battery   (P_bat > 0)
      E_in  (kWh) : energy recovered by braking     (P_bat < 0)
    """
    
    J_TO_KWH = 1 / 3.6e6
    
    distance = np.trapezoid(speed * KMH_TO_MS, t) * 1e-3
    E_out = np.trapezoid(np.where(P_bat > 0, P_bat, 0), t) * J_TO_KWH
    E_in = np.trapezoid(np.where(P_bat < 0, P_bat, 0), t) * J_TO_KWH
    
    return distance, E_out, E_in



