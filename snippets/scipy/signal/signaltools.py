import math
from ulab import numpy
from ulab import numpy as np
from ulab import scipy as spy

from ...numpy import (atleast_1d, atleast_2d, poly, asarray, prod, size, append, nonzero, zeros_like, isreal, moveaxis)
from .filter_design import _validate_sos
from ..linalg import companion
from ._arraytools import (axis_slice, axis_reverse, even_ext, odd_ext, const_ext)
from ._sosfilt import _sosfilt

def _validate_x(x):
    x = asarray(x)
    if len(x.shape) == 0:
        raise ValueError('x must be at least 1-D')
    return x

def _validate_pad(padtype, padlen, x, axis, ntaps):
    """Helper to validate padding for filtfilt"""
    if padtype not in ['even', 'odd', 'constant', None]:
        raise ValueError(("Unknown value '%s' given to padtype.  padtype "
                          "must be 'even', 'odd', 'constant', or None.") %
                         padtype)

    if padtype is None:
        padlen = 0

    if padlen is None:
        # Original padding; preserved for backwards compatibility.
        edge = ntaps * 3
    else:
        edge = padlen

    # x's 'axis' dimension must be bigger than edge.
    if x.shape[axis] <= edge:
        raise ValueError("The length of the input vector x must be greater "
                         "than padlen, which is %d." % edge)

    if padtype is not None and edge > 0:
        # Make an extension of length `edge` at each
        # end of the input array.
        if padtype == 'even':
            ext = even_ext(x, edge, axis=axis)
        elif padtype == 'odd':
            ext = odd_ext(x, edge, axis=axis)
        else:
            ext = const_ext(x, edge, axis=axis)
    else:
        ext = x
    return edge, ext

def lfilter_zi(b, a):
    """
    Construct initial conditions for lfilter for step response steady-state.
    Compute an initial state `zi` for the `lfilter` function that corresponds
    to the steady state of the step response.
    A typical use of this function is to set the initial state so that the
    output of the filter starts at the same value as the first element of
    the signal to be filtered.
    Parameters
    ----------
    b, a : array_like (1-D)
        The IIR filter coefficients. See `lfilter` for more
        information.
    Returns
    -------
    zi : 1-D ndarray
        The initial state for the filter.
    See Also
    --------
    lfilter, lfiltic, filtfilt
    Notes
    -----
    A linear filter with order m has a state space representation (A, B, C, D),
    for which the output y of the filter can be expressed as::
        z(n+1) = A*z(n) + B*x(n)
        y(n)   = C*z(n) + D*x(n)
    where z(n) is a vector of length m, A has shape (m, m), B has shape
    (m, 1), C has shape (1, m) and D has shape (1, 1) (assuming x(n) is
    a scalar).  lfilter_zi solves::
        zi = A*zi + B
    In other words, it finds the initial condition for which the response
    to an input of all ones is a constant.
    Given the filter coefficients `a` and `b`, the state space matrices
    for the transposed direct form II implementation of the linear filter,
    which is the implementation used by scipy.signal.lfilter, are::
        A = scipy.linalg.companion(a).T
        B = b[1:] - a[1:]*b[0]
    assuming `a[0]` is 1.0; if `a[0]` is not 1, `a` and `b` are first
    divided by a[0].
    Examples
    --------
    The following code creates a lowpass Butterworth filter. Then it
    applies that filter to an array whose values are all 1.0; the
    output is also all 1.0, as expected for a lowpass filter.  If the
    `zi` argument of `lfilter` had not been given, the output would have
    shown the transient signal.
    >>> from numpy import array, ones
    >>> from scipy.signal import lfilter, lfilter_zi, butter
    >>> b, a = butter(5, 0.25)
    >>> zi = lfilter_zi(b, a)
    >>> y, zo = lfilter(b, a, ones(10), zi=zi)
    >>> y
    array([1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
    Another example:
    >>> x = array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
    >>> y, zf = lfilter(b, a, x, zi=zi*x[0])
    >>> y
    array([ 0.5       ,  0.5       ,  0.5       ,  0.49836039,  0.48610528,
        0.44399389,  0.35505241])
    Note that the `zi` argument to `lfilter` was computed using
    `lfilter_zi` and scaled by `x[0]`.  Then the output `y` has no
    transient until the input drops from 0.5 to 0.0.
    """
    # FIXME: Can this function be replaced with an appropriate
    # use of lfiltic?  For example, when b,a = butter(N,Wn),
    #    lfiltic(b, a, y=numpy.ones_like(a), x=numpy.ones_like(b)).
    #

    # We could use scipy.signal.normalize, but it uses warnings in
    # cases where a ValueError is more appropriate, and it allows
    # b to be 2D.
    b = atleast_1d(b)
    if len(b.shape) != 1:
        raise ValueError("Numerator b must be 1-D.")
    a = atleast_1d(a)
    if len(a.shape) != 1:
        raise ValueError("Denominator a must be 1-D.")

    while len(a) > 1 and a[0] == 0.0:
        a = a[1:]
    if a.size < 1:
        raise ValueError("There must be at least one nonzero `a` coefficient.")

    if a[0] != 1.0:
        # Normalize the coefficients so a[0] == 1.
        b = b / a[0]
        a = a / a[0]

    n = max(len(a), len(b))

    # Pad a or b with zeros so they are the same length.
    if len(a) < n:
        a = np.r_[a, np.zeros(n - len(a))]
    elif len(b) < n:
        b = np.r_[b, np.zeros(n - len(b))]

    #IminusA = np.eye(n - 1, dtype=np.result_type(a, b)) - companion(a).T
    IminusA = np.eye(n - 1, dtype=np.float) - companion(a).T

    B = b[1:] - a[1:] * b[0]
    # Solve zi = A*zi + B
    #zi = np.linalg.solve(IminusA, B)

    # For future reference: we could also use the following
    # explicit formulas to solve the linear system:
    #
    zi = np.zeros(n - 1)
    zi[0] = np.sum(B) / np.sum(IminusA[:,0])
    asum = 1.0
    csum = 0.0
    for k in range(1,n-1):
        asum += a[k]
        csum += b[k] - a[k]*b[0]
        zi[k] = asum*zi[0] - csum

    return zi

def sosfilt_zi(sos):
    """
    Construct initial conditions for sosfilt for step response steady-state.
    Compute an initial state `zi` for the `sosfilt` function that corresponds
    to the steady state of the step response.
    A typical use of this function is to set the initial state so that the
    output of the filter starts at the same value as the first element of
    the signal to be filtered.
    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. See `sosfilt` for the SOS filter format
        specification.
    Returns
    -------
    zi : ndarray
        Initial conditions suitable for use with ``sosfilt``, shape
        ``(n_sections, 2)``.
    See Also
    --------
    sosfilt, zpk2sos
    Notes
    -----
    .. versionadded:: 0.16.0
    Examples
    --------
    Filter a rectangular pulse that begins at time 0, with and without
    the use of the `zi` argument of `scipy.signal.sosfilt`.
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> sos = signal.butter(9, 0.125, output='sos')
    >>> zi = signal.sosfilt_zi(sos)
    >>> x = (np.arange(250) < 100).astype(int)
    >>> f1 = signal.sosfilt(sos, x)
    >>> f2, zo = signal.sosfilt(sos, x, zi=zi)
    >>> plt.plot(x, 'k--', label='x')
    >>> plt.plot(f1, 'b', alpha=0.5, linewidth=2, label='filtered')
    >>> plt.plot(f2, 'g', alpha=0.25, linewidth=4, label='filtered with zi')
    >>> plt.legend(loc='best')
    >>> plt.show()
    """
    sos = asarray(sos)
    if len(sos.shape) != 2 or sos.shape[1] != 6:
        raise ValueError('sos must be shape (n_sections, 6)')

    #if sos.dtype.kind in 'bui':
    #    sos = sos.astype(np.float)

    n_sections = sos.shape[0]
    zi = np.empty((n_sections, 2), dtype=sos.dtype)
    scale = 1.0
    for section in range(n_sections):
        b = sos[section, :3]
        a = sos[section, 3:]
        zi[section] = scale * lfilter_zi(b, a)
        # If H(z) = B(z)/A(z) is this section's transfer function, then
        # b.sum()/a.sum() is H(1), the gain at omega=0.  That's the steady
        # state value of this section's step response.
        scale *= np.sum(b) / np.sum(a)

    return zi

def sosfilt(sos, x, axis=-1, zi=None):
    """
    Filter data along one dimension using cascaded second-order sections.
    Filter a data sequence, `x`, using a digital IIR filter defined by
    `sos`.
    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.
    x : array_like
        An N-dimensional input array.
    axis : int, optional
        The axis of the input data array along which to apply the
        linear filter. The filter is applied to each subarray along
        this axis.  Default is -1.
    zi : array_like, optional
        Initial conditions for the cascaded filter delays.  It is a (at
        least 2D) vector of shape ``(n_sections, ..., 2, ...)``, where
        ``..., 2, ...`` denotes the shape of `x`, but with ``x.shape[axis]``
        replaced by 2.  If `zi` is None or is not given then initial rest
        (i.e. all zeros) is assumed.
        Note that these initial conditions are *not* the same as the initial
        conditions given by `lfiltic` or `lfilter_zi`.
    Returns
    -------
    y : ndarray
        The output of the digital filter.
    zf : ndarray, optional
        If `zi` is None, this is not returned, otherwise, `zf` holds the
        final filter delay values.
    See Also
    --------
    zpk2sos, sos2zpk, sosfilt_zi, sosfiltfilt, sosfreqz
    Notes
    -----
    The filter function is implemented as a series of second-order filters
    with direct-form II transposed structure. It is designed to minimize
    numerical precision errors for high-order filters.
    .. versionadded:: 0.16.0
    Examples
    --------
    Plot a 13th-order filter's impulse response using both `lfilter` and
    `sosfilt`, showing the instability that results from trying to do a
    13th-order filter in a single stage (the numerical error pushes some poles
    outside of the unit circle):
    >>> import matplotlib.pyplot as plt
    >>> from scipy import signal
    >>> b, a = signal.ellip(13, 0.009, 80, 0.05, output='ba')
    >>> sos = signal.ellip(13, 0.009, 80, 0.05, output='sos')
    >>> x = signal.unit_impulse(700)
    >>> y_tf = signal.lfilter(b, a, x)
    >>> y_sos = signal.sosfilt(sos, x)
    >>> plt.plot(y_tf, 'r', label='TF')
    >>> plt.plot(y_sos, 'k', label='SOS')
    >>> plt.legend(loc='best')
    >>> plt.show()
    """
    x = _validate_x(x)
    sos, n_sections = _validate_sos(sos)
    x_zi_shape = list(x.shape)
    x_zi_shape[axis] = 2
    x_zi_shape = tuple([n_sections] + x_zi_shape)
    inputs = [sos, x]
    if zi is not None:
        inputs.append(np.asarray(zi))
    dtype = np.result_type(*inputs)
    if dtype.char not in 'fdgFDGO':
        raise NotImplementedError("input type '%s' not supported" % dtype)
    if zi is not None:
        zi = np.array(zi, dtype)  # make a copy so that we can operate in place
        if zi.shape != x_zi_shape:
            raise ValueError('Invalid zi shape. With axis=%r, an input with '
                             'shape %r, and an sos array with %d sections, zi '
                             'must have shape %r, got %r.' %
                             (axis, x.shape, n_sections, x_zi_shape, zi.shape))
        return_zi = True
    else:
        zi = np.zeros(x_zi_shape, dtype=dtype)
        return_zi = False
    axis = axis % x.ndim  # make positive
    x = moveaxis(x, axis, -1)
    zi = moveaxis(zi, [0, axis + 1], [-2, -1])
    x_shape, zi_shape = x.shape, zi.shape
    x = np.reshape(x, (-1, x.shape[-1]))
    x = np.array(x, dtype, order='C')  # make a copy, can modify in place
    zi = np.ascontiguousarray(np.reshape(zi, (-1, n_sections, 2)))
    sos = sos.astype(dtype, copy=False)
    _sosfilt(sos, x, zi)
    x.shape = x_shape
    x = moveaxis(x, -1, axis)
    if return_zi:
        zi.shape = zi_shape
        zi = moveaxis(zi, [-2, -1], [0, axis + 1])
        out = (x, zi)
    else:
        out = x
    return out

def sosfiltfilt(sos, x, axis=-1, padtype='odd', padlen=None):
    """
    A forward-backward digital filter using cascaded second-order sections.
    See `filtfilt` for more complete information about this method.
    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.
    x : array_like
        The array of data to be filtered.
    axis : int, optional
        The axis of `x` to which the filter is applied.
        Default is -1.
    padtype : str or None, optional
        Must be 'odd', 'even', 'constant', or None.  This determines the
        type of extension to use for the padded signal to which the filter
        is applied.  If `padtype` is None, no padding is used.  The default
        is 'odd'.
    padlen : int or None, optional
        The number of elements by which to extend `x` at both ends of
        `axis` before applying the filter.  This value must be less than
        ``x.shape[axis] - 1``.  ``padlen=0`` implies no padding.
        The default value is::
            3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(),
                                        (sos[:, 5] == 0).sum()))
        The extra subtraction at the end attempts to compensate for poles
        and zeros at the origin (e.g. for odd-order filters) to yield
        equivalent estimates of `padlen` to those of `filtfilt` for
        second-order section filters built with `scipy.signal` functions.
    Returns
    -------
    y : ndarray
        The filtered output with the same shape as `x`.
    See Also
    --------
    filtfilt, sosfilt, sosfilt_zi, sosfreqz
    Notes
    -----
    .. versionadded:: 0.18.0
    Examples
    --------
    >>> from scipy.signal import sosfiltfilt, butter
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    Create an interesting signal to filter.
    >>> n = 201
    >>> t = np.linspace(0, 1, n)
    >>> x = 1 + (t < 0.5) - 0.25*t**2 + 0.05*rng.standard_normal(n)
    Create a lowpass Butterworth filter, and use it to filter `x`.
    >>> sos = butter(4, 0.125, output='sos')
    >>> y = sosfiltfilt(sos, x)
    For comparison, apply an 8th order filter using `sosfilt`.  The filter
    is initialized using the mean of the first four values of `x`.
    >>> from scipy.signal import sosfilt, sosfilt_zi
    >>> sos8 = butter(8, 0.125, output='sos')
    >>> zi = x[:4].mean() * sosfilt_zi(sos8)
    >>> y2, zo = sosfilt(sos8, x, zi=zi)
    Plot the results.  Note that the phase of `y` matches the input, while
    `y2` has a significant phase delay.
    >>> plt.plot(t, x, alpha=0.5, label='x(t)')
    >>> plt.plot(t, y, label='y(t)')
    >>> plt.plot(t, y2, label='y2(t)')
    >>> plt.legend(framealpha=1, shadow=True)
    >>> plt.grid(alpha=0.25)
    >>> plt.xlabel('t')
    >>> plt.show()
    """
    sos, n_sections = _validate_sos(sos)
    x = _validate_x(x)

    np.set_printoptions(threshold=1000)

    # `method` is "pad"...
    ntaps = 2 * n_sections + 1
    a = np.sum((sos[:, 2] == 0))
    b = np.sum(sos[:, 5] == 0)
    ntaps -= min(a,b)

    edge, ext = _validate_pad(padtype, padlen, x, axis,
                              ntaps=ntaps)

    # These steps follow the same form as filtfilt with modifications
    zi = sosfilt_zi(sos)  # shape (n_sections, 2) --> (n_sections, ..., 2, ...)

    zi_shape = [1] * len(x.shape)
    zi_shape[axis] = 2
    zi.shape = tuple([n_sections] + zi_shape)
    x_0 = axis_slice(ext, stop=1, axis=axis)  

    (y, zf) = spy.signal.sosfilt(sos, ext, zi=zi* x_0)
    y_0 = axis_slice(y, start=-1, axis=axis)
    (y, zf) = spy.signal.sosfilt(sos, axis_reverse(y, axis=axis), zi=zi * y_0)
   
    y = axis_reverse(y, axis=axis)
    if edge > 0:
        y = axis_slice(y, start=edge, stop=-edge, axis=axis)
    return y