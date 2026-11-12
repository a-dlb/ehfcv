#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 12:38:47 2026

@author: antoinedlbck
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vehicle import Vehicle
from motor import Motor
from constants import (KMH_TO_MS, RADS_TO_RPM, P_ACCESSORIES, 
                       ALPHA_REGEN, P_REGEN_MAX, BATTERY_CAPACITY)
from resistive_forces import total_force
from plot_style import get_style
from plots import plot_consumption, plot_cycle

PLOT = True
EXPORT = False
WLTP_FILE = "WLTP.xlsx" # Excel 

#-------------------------------- affichage titre -------------------------------------
def print_section(title):
    print("\n" + "=" * 30)
    print(f"{title.upper()}") 
    print("=" * 30)
    return 


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

def battery_power_constant_speed(vehicle, motor, speed, slope=0):
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
    
    return F_t * speed_ms / (eta_tr * eta_m) + P_ACCESSORIES


def consumption_constant_speed(vehicle, motor, speed, slope=0):
    """
    Energy consumption E (kWh/km) at constant speed
    (!) speed is expected in (km/h) (and slope in (%))
    
      E = P_bat (kW) / V (km/h)
    """
    
    P_bat = battery_power_constant_speed(vehicle, motor, speed, slope)
    
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
    
    distance = np.trapz(speed * KMH_TO_MS, t) * 1e-3
    E_out = np.trapz(np.where(P_bat > 0, P_bat, 0), t) * J_TO_KWH
    E_in = np.trapz(np.where(P_bat < 0, P_bat, 0), t) * J_TO_KWH
    
    return distance, E_out, E_in



#-------------------------------- affichage valeurs -------------------------------------
def main():
    car = Vehicle()
    motor = Motor()
    
    # a. Energy consumption at constant speed
    print_section("Consumption at constant speed")
    speeds = np.array([30, 50, 90, 120])           # km/h
    rpm, C_m = motor_operating_point(car, speeds)
    eta_m = motor_efficiency(motor, rpm, C_m) 
    P_bat = battery_power_constant_speed(car, motor, speeds)
    E = consumption_constant_speed(car, motor, speeds)
    
    print(f"{'V (km/h)':>10}{'N (rpm)':>10}{'C_m (Nm)':>10}{'eta_m (%)':>11}"
          f"{'P_bat (kW)':>12}{'E (kWh/km)':>12}")
    print("-" * 65)
    
    for v, N, C, eta, P, e in zip(speeds, rpm, C_m, eta_m, P_bat, E):
        print(f"{v:>10d}{N:>10.0f}{C:>10.2f}{eta * 1e2:>11.1f}"
              f"{P * 1e-3:>12.2f}{e:>12.4f}")
      
        
    # b. Plots
    print_section("Influence of the slope")
    slopes = [0, 5, 10] # %, Influence of the slope
    
    print(f"{'V (km/h)':>10}" + "".join(f"{f'E at {s} % (kWh/km)':>24}" for s in slopes))
    print("-" * 82)
    
    for v in speeds:
        print(f"{v:>10d}" + "".join(
            f"{consumption_constant_speed(car, motor, v, slope=s):>24.4f}" for s in slopes))
    
    
    if PLOT:
        speed = np.linspace(5, 150, 1000) # km/h
        E_slopes = [consumption_constant_speed(car, motor, speed, slope=s) 
            for s in slopes]

        print("\nMinimum energy consumption:")
        for s, E_s in zip(slopes, E_slopes):
            k = np.argmin(E_s) # index of the smallest consumption
            print(f"  {s:>2d} % slope : E_min = {E_s[k]:.4f} (kWh/km) at V = {speed[k]:.1f} (km/h)")

     # c. WLTP driving cycle
    print_section("WLTP driving cycle")
    t, v_wltp, phase = load_wltp(WLTP_FILE)
    P_wltp = battery_power_cycle(car, motor, t, v_wltp)
    
    print(f"{'Phase':>12}{'d (km)':>10}{'E_out (kWh)':>13}{'E_in (kWh)':>12}"
          f"{'E_net (kWh)':>13}{'E (kWh/km)':>12}")
    print("-" * 72)
    
    # One row per phase of the cycle
    for name in pd.unique(phase): # phases in the order of the cycle
        mask = phase == name
        d, E_out, E_in = cycle_energy(t[mask], v_wltp[mask], P_wltp[mask])
        print(f"{name:>12}{d:>10.2f}{E_out:>13.3f}{E_in:>12.3f}"
              f"{E_out + E_in:>13.3f}{(E_out + E_in) / d:>12.4f}")
    
    # Whole cycle
    d, E_out, E_in = cycle_energy(t, v_wltp, P_wltp)
    E_net = E_out + E_in
    E_wltp = E_net / d # kWh/km
    
    print("-" * 72)
    print(f"{'Total':>12}{d:>10.2f}{E_out:>13.3f}{E_in:>12.3f}"
          f"{E_net:>13.3f}{E_wltp:>12.4f}")
    
    print(f"\n  Duration: {t[-1] - t[0]:.0f} (s) ; mean speed: {d / (t[-1] - t[0]) * 3600:.1f} (km/h) ; "
      f"maximum speed: {v_wltp.max():.1f} (km/h)")
    print(f"  Maximum battery power: {P_wltp.max() * 1e-3:.1f} (kW) ; "
      f"maximum recovered power: {-P_wltp.min() * 1e-3:.1f} (kW)")
    # Share of the cycle where the battery delivers less than 10 kW
    P_low = 10e3 # W
    share = (P_wltp < P_low).mean() * 1e2 # %
    print(f"  Battery power below {P_low * 1e-3:.0f} kW during {share:.1f} % of the cycle") 

    # Mean powers over the cycle (kW): energy (kWh) / duration (h)
    T_h = (t[-1] - t[0]) / 3600 # duration of the cycle (h)
    print(f"  Mean battery power: out = {E_out / T_h:.2f} (kW) ; "
      f"in = {E_in / T_h:.2f} (kW) ; net = {E_net / T_h:.2f} (kW)")
    print(f"  Recovered / drawn: {-E_in / E_out * 1e2:.1f} %")

    print(f"  Energy consumption: {E_wltp:.4f} (kWh/km)")
    print(f"  Range (battery of {BATTERY_CAPACITY} kWh): {BATTERY_CAPACITY / E_wltp:.1f} (km)")
    # Consumption at a constant speed equal to the mean speed of the cycle
    v_mean = d / (t[-1] - t[0]) * 3600 # km/h
    E_mean = consumption_constant_speed(car, motor, v_mean)
    print(f"  Consumption at a constant speed of {v_mean:.1f} km/h: {E_mean:.4f} (kWh/km) ")
    
    
    if PLOT:
       with plt.rc_context(get_style()):
           path = "Graphs/wltp.pdf" if EXPORT else None
           phase_plot = np.where(phase == "Middle", "Medium", phase) 
           plot_cycle(t, v_wltp, P_wltp, phase=phase_plot, export_path=path)
       plt.show()
    return

if __name__ == "__main__":
    
    main()
