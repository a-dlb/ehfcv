#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 15:13:46 2026

@author: antoinedlbck
"""

def get_style(style="default"):
    
    if style == "default":
        
        style_dict = {
            
        "font.family": "sans-serif", 
        "font.sans-serif": "Tahoma",
        "mathtext.fontset": "dejavuserif",
        "xtick.labelsize": 30,  
        "ytick.labelsize": 30,
        "xtick.major.pad": 20,
        "ytick.major.pad": 20,
        'font.size': 35,
        "figure.figsize": (15, 12),
        "lines.linewidth": 6,
        "axes.labelpad": 15,
        "savefig.format": 'pdf',
        "grid.linewidth": 1,
        "xtick.major.size": 10,
        "ytick.major.size": 10,
        "legend.fontsize": 26,
        "figure.constrained_layout.use": True,
        "lines.markersize": 20,
        
        }
    
    # other style dictionnary for the max slope bar chart otherwise it's crappy
    elif style == "maxslope":
    
        style_dict = get_style("default")
        style_dict.update({

        "xtick.labelsize": 25,
        "ytick.labelsize": 25,
        "xtick.major.pad": 8,
        "ytick.major.pad": 8,
        "font.size": 28,
        "figure.figsize": (22, 12),
        "lines.linewidth": 4,
        "axes.labelpad": 10,
        "legend.fontsize": 30,

        })
    
    return style_dict