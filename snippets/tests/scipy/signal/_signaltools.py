import math
import sys
sys.path.append('.')

from snippets import scipy
from ulab import numpy as np

np.set_printoptions(threshold=100)

z = np.array([1.0+0.0j, 1.0+0.0j, 1.0+0.0j, 1.0+0.0j, -1.0+0.0j, -1.0+0.0j], dtype=np.complex) 
p = np.array([0.9973072, 0.9973072, 0.1122792, 0.1122792, 0.9860069, 0.9078636]) 
k = 0.2343006
print(scipy.zpk2sos(z,p,k))

wave_duration = 3
sample_rate = 100
freq = 2
q = 5
samples = wave_duration*sample_rate
samples_decimated = int(samples/q)
x = np.linspace(0, wave_duration, samples, endpoint=False)
y = np.cos(x*np.pi*freq*2)
print(y)
ydem = scipy.decimate(y, q)

print(ydem)