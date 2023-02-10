import numpy as np
import numba
from numba import njit, float64, void, bool_    # import the types
from numba.experimental import jitclass
from numba.typed import List

spec = [
    ('x', float64),
    ('y', float64),
    ('z', float64),
    ('data', float64[:]),
    ('mask', bool_[:])
]


@jitclass(spec)
class Struct:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.data = np.zeros(1)
        self.mask = np.array([True])

@njit
def jit_test(tracers, idx):
    r = 0.
    for i in idx:
    # for t in tracers:
        t = tracers[i]
        print(i)
        c = t.x**2 + t.y**2 + t.z**2
        r += c * np.sum(t.data)
    
    return r

# @njit
def test():
    tracers = np.empty(10, dtype=Struct)
    for i in range(10):
        d = i * np.arange(10, dtype=np.float64)
        x = np.float64(i)
        tracers[i] = Struct(x, x**2, x**3)
        tracers[i].data = d
        tracers[i].mask = np.array([True, False, True, False])

    l = List()
    [l.append(x) for x in tracers]
    # mask = np.array([True for i in range(10)])
    # l[mask]
    idx = np.array([1, 3, 7])
    print(jit_test(l, idx))
    # return tracers

