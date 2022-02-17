import math
import sys
sys.path.append('.')

from snippets import scipy
from ulab import numpy as np

np.set_printoptions(threshold=100)

print(scipy.companion([1, -10, 31, -30]))
