
def _sosfilt_float(sos, x, zi):
    # Modifies x and zi in place
    n_signals = x.shape[0]
    n_samples = x.shape[1]
    n_sections = sos.shape[0]

    # jumping through a few memoryview hoops to reduce array lookups,
    # the original version is still in the gil version below.
    for i in xrange(n_signals):
        zi_slice = zi[i, :, :]
        for n in xrange(n_samples):

            x_cur = x[i, n]

            for s in xrange(n_sections):
                x_new = sos[s, 0] * x_cur + zi_slice[s, 0]
                zi_slice[s, 0] = (sos[s, 1] * x_cur - sos[s, 4] * x_new
                                  + zi_slice[s, 1])
                zi_slice[s, 1] = sos[s, 2] * x_cur - sos[s, 5] * x_new
                x_cur = x_new

            x[i, n] = x_cur


def _sosfilt_object(sos, x, zi):
    # Modifies x and zi in place
    n_signals = x.shape[0]
    n_samples = x.shape[1]
    n_sections = sos.shape[0]

    for i in xrange(n_signals):
        for n in xrange(n_samples):
            for s in xrange(n_sections):
                x_n = x[i, n]  # make a temporary copy
                # Use direct II transposed structure:
                x[i, n] = sos[s, 0] * x_n + zi[i, s, 0]
                zi[i, s, 0] = (sos[s, 1] * x_n - sos[s, 4] * x[i, n] +
                               zi[i, s, 1])
                zi[i, s, 1] = (sos[s, 2] * x_n - sos[s, 5] * x[i, n])


def _sosfilt(sos, x, zi):
    if type(x) is object:
        return _sosfilt_object(sos, x, zi)
    else:
        return _sosfilt_float(sos, x, zi)

