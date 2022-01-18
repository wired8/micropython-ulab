from ulab import numpy as np
from ...numpy import (atleast_1d, atleast_2d, poly, asarray, prod, size, append, nonzero, zeros_like, isreal, moveaxis)

def companion(a):
    """
    Create a companion matrix.
    Create the companion matrix [1]_ associated with the polynomial whose
    coefficients are given in `a`.
    Parameters
    ----------
    a : (N,) array_like
        1-D array of polynomial coefficients. The length of `a` must be
        at least two, and ``a[0]`` must not be zero.
    Returns
    -------
    c : (N-1, N-1) ndarray
        The first row of `c` is ``-a[1:]/a[0]``, and the first
        sub-diagonal is all ones.  The data-type of the array is the same
        as the data-type of ``1.0*a[0]``.
    Raises
    ------
    ValueError
        If any of the following are true: a) ``a.ndim != 1``;
        b) ``a.size < 2``; c) ``a[0] == 0``.
    Notes
    -----
    .. versionadded:: 0.8.0
    References
    ----------
    .. [1] R. A. Horn & C. R. Johnson, *Matrix Analysis*.  Cambridge, UK:
        Cambridge University Press, 1999, pp. 146-7.
    Examples
    --------
    >>> from scipy.linalg import companion
    >>> companion([1, -10, 31, -30])
    array([[ 10., -31.,  30.],
           [  1.,   0.,   0.],
           [  0.,   1.,   0.]])
    """
    a = atleast_1d(a)

    if len(a.shape) != 1:
        raise ValueError("Incorrect shape for `a`.  `a` must be "
                         "one-dimensional.")

    if a.size < 2:
        raise ValueError("The length of `a` must be at least 2.")

    if a[0] == 0:
        raise ValueError("The first coefficient in `a` must not be zero.")

    first_row = -a[1:] / (1.0 * a[0])
    n = a.size
    c = np.zeros((n - 1, n - 1), dtype=first_row.dtype)
    c[0] = first_row

    la = list(range(1, n - 1))
    lb = list(range(0, n - 2))

    if len(la) > 1:
         c[la[1],la[0]] = 1
         c[lb[1],lb[0]] = 1
    else:
         c[la[0],lb[0]] = 1

    return c
