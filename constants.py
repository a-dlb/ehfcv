#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:40:20 2026

@author: antoinedlbck
"""

import numpy as np

G = 9.81 # m/s^2 
INCH_TO_M = 25.4e-3 # m/inch
GEAR_PAIR_EFF = 0.9875 # -
RPM_TO_RADS = 2 * np.pi / 60 # rad/rpm
RADS_TO_RPM = 60 / (2 * np.pi) # rpm/rad
MS_TO_KMH = 3.6 # (km/h)/(m/s)
KMH_TO_MS = 1/3.6 # (m/s)/(km/h)
RHO_AIR = 1.23 # kg/m^3