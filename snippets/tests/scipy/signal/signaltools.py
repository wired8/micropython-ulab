import math
import sys
sys.path.append('.')

from snippets.random import random
from snippets import scipy
from ulab import numpy as np

np.set_printoptions(threshold=1000)

 
rng = random.randrange(1)
n = 201
t = np.linspace(0, 1, n)
x = 1 + (t < 0.5) - 0.25*t**2 + 0.05*rng

sos = scipy.butter(4, 0.125, output='sos')

y = scipy.sosfiltfilt(sos, x)
print (y)