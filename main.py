import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from utils.Dispersion_relation import dispersion_relation
from utils.Hough_and_derivatives import hough_and_derivatives
from utils.Dynamic_three_waves import triad_evolution

#dispersion_relation(10000)
#hough_and_derivatives(1,2,3,10000)
triad_evolution(10000, 1,1,3,3,4,3,4,5,3,10,10,10,0,100)