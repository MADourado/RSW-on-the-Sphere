import argparse
import yaml
import os

import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from utils.Dispersion_relation import dispersion_relation
from utils.Hough_and_derivatives import hough_and_derivatives
from utils.Dynamic_three_waves import triad_evolution

def load_config(path):

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs.yaml",
        help="You should provide a config via some .yaml. See the default configs.yaml to help!"
    )
    args = parser.parse_args()
    config = load_config(args.config)

    output = config['OUTPUT_PATH']
    os.makedirs(output, exist_ok=True)

    h_e = config['h_e']

    if config['dispersion_relation']:
        dispersion_relation(h_e, f'{output}/dispersion_relation.png')
    
    print()
    print('Starting Triad')
    print()
    m_a, n_a, alpha_a = config['Triad']['mode_a']['m'],config['Triad']['mode_a']['n'],config['Triad']['mode_a']['alpha']
    m_b, n_b, alpha_b = config['Triad']['mode_b']['m'],config['Triad']['mode_b']['n'],config['Triad']['mode_b']['alpha']
    m_c, n_c, alpha_c = config['Triad']['mode_c']['m'],config['Triad']['mode_c']['n'],config['Triad']['mode_c']['alpha']

    if config['Triad']['mode_a']['show_mode']:
        os.makedirs(f'{output}/mode_a/', exist_ok=True)
        hough_and_derivatives(m_a,n_a,alpha_a,h_e, f'{output}/mode_a/')
    if config['Triad']['mode_b']['show_mode']:
        os.makedirs(f'{output}/mode_b/', exist_ok=True)
        hough_and_derivatives(m_b,n_b,alpha_b,h_e, f'{output}/mode_b/')
    if config['Triad']['mode_c']['show_mode']:
        os.makedirs(f'{output}/mode_c/', exist_ok=True)
        hough_and_derivatives(m_c,n_c,alpha_c,h_e, f'{output}/mode_c/')
    
    print('Triad finished')
    print()
    print('Starting Dynamics')
    print()
    t0 = config['Dynamics']['t0']
    tf = config['Dynamics']['tf']*4*np.pi
    h  = config['Dynamics']['h']

    u_a = config['Triad']['mode_a']['u']
    u_b = config['Triad']['mode_b']['u']
    u_c = config['Triad']['mode_c']['u']

    if config['Dynamics']['show_dynamics']:
        triad_evolution(h_e, m_a, n_a, alpha_a, m_b, n_b, 
                 alpha_b, m_c,n_c, alpha_c, u_a, u_b, u_c, t0, tf,h, f'{output}/dynamics.png')

    print('Dynamics finished')

if __name__ == "__main__":
    main()