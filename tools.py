# Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU
# General Public License. The full license is in the file LICENSE,
# distributed with this software.

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.ndimage


def cross_product(ux, uy, vx, vy):
    return uy * vx - ux * vy


def cubic_pulse(npoints):
    r = np.abs(np.linspace(-1, 1, npoints))
    return 1.0 - r * r * (3.0 - 2.0 * r)


def cumulative_distance(x, y, normalized=True):
    xy = np.array((x, y)).T
    distance = np.cumsum(np.sqrt(np.sum(np.diff(xy, axis=0)**2, axis=1)))
    distance = np.insert(distance, 0, 0)
    if normalized:
        distance = distance / distance[-1]
    return distance


# compute the local curvature
def curvature(x, y, normalized=True):
    xs = diff_ds(x)
    ys = diff_ds(y)
    xss = diff_ds2(x)
    yss = diff_ds2(y)
    kappa = np.abs(xs * yss - ys * xss) / np.power(xs * xs + ys * ys, 1.5)

    if normalized:
        return kappa / np.max(kappa)
    else:
        return kappa


# 1st derivate dx / ds
def diff_ds(x):
    ds = np.zeros_like(x)
    ds[1:-1] = 0.5 * (x[2:] - x[:-2])
    ds[0] = x[1] - x[0]
    ds[-1] = x[-1] - x[-2]
    return ds


# 2nd derivative d2x / ds2
def diff_ds2(x):
    ds = np.zeros_like(x)
    ds[1:-1] = x[:-2] - 2 * x[1:-1] + x[2:]
    ds[0] = x[1] - x[0]
    ds[-1] = x[-1] - x[-2]
    return ds


def interp_spline(x, y, npoints=200, method='cubic'):
    distance = cumulative_distance(x, y)
    fitp = scipy.interpolate.interp1d(distance,
                                      np.array((x, y)).T,
                                      kind=method,
                                      axis=0)
    xyi = fitp(np.linspace(0, 1, npoints))
    return xyi[:, 0], xyi[:, 1]


# Return true if line segments AB and CD intersect
# https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
def intersect(A, B, C, D):

    def ccw(A_, B_, C_):
        return (C_[1] - A_[1]) * (B_[0] - A_[0]) > (B_[1] - A_[1]) * (C_[0] -
                                                                      A_[0])

    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def nearest_neighbor_search_2d(x, y, i_start=None):
    if i_start is None:
        i_start = np.argmax(x**2 + y**2)

    xp = [x[i_start]]
    yp = [y[i_start]]

    x = np.delete(x, i_start)
    y = np.delete(y, i_start)

    while len(x) > 0:
        r2 = (x - xp[-1])**2 + (y - yp[-1])**2
        i = np.argmin(r2)
        xp.append(x[i])
        yp.append(y[i])

        x = np.delete(x, i)
        y = np.delete(y, i)

    x = np.array(xp)
    y = np.array(yp)

    return np.array(xp), np.array(yp)


def meander_midpoint(x, y, amp, iterations=1):

    for _ in range(iterations):
        xm = []
        ym = []

        # decide 1st midpoint displacement based on the path curvature
        cp = cross_product(x[1] - x[0], y[1] - y[0], x[2] - x[0], y[2] - y[0])
        cp = np.sign(cp)

        for k in range(x.size - 1):
            angle = np.arctan2(y[k + 1] - y[k], x[k + 1] - x[k])
            dist = np.hypot(x[k + 1] - x[k], y[k + 1] - y[k])

            xmid = 0.5 * (x[k + 1] + x[k])
            ymid = 0.5 * (y[k + 1] + y[k])

            da = cp * np.pi / 2
            xmid += amp * dist * np.cos(angle + da)
            ymid += amp * dist * np.sin(angle + da)

            xm += [x[k], xmid]
            ym += [y[k], ymid]

            # alternate mid-point displacement sign
            cp *= -1

        xm.append(x[-1])
        ym.append(y[-1])

        x = np.array(xm)
        y = np.array(ym)

    return x, y


def meander_strengthen(x,
                       y,
                       iterations,
                       ir,
                       tangent_ratio,
                       normal_ratio,
                       npoints_mini,
                       curvature_normalization=1):

    loops_x = []
    loops_y = []

    for _ in range(iterations):

        # factor applied to the deformation: beginning and end of the
        # curve are not modified to avoid numerical instabilities
        shape_factor = tricube(x.size)

        kappa = curvature(x, y, False) / curvature_normalization

        #
        dx = np.zeros_like(x)
        dy = np.zeros_like(y)

        for k in range(x.size):
            # curve local angle
            if (k < x.size - 1):
                angle = np.arctan2(y[k + 1] - y[k], x[k + 1] - x[k])
            else:
                angle = np.arctan2(y[k] - y[k - 1], x[k] - x[k - 1])

            # curve "orientation"
            if (k < x.size - 2):
                cp = cross_product(x[k + 1] - x[k], y[k + 1] - y[k],
                                   x[k + 2] - x[k], y[k + 2] - y[k])
            else:
                cp = 1

            # normal vector
            nx = np.cos(angle)
            ny = np.sin(angle)

            # tangent vector (whose orientation is driven by the
            # cross-product defined above)
            tx = np.cos(angle + np.sign(cp) * np.pi / 2)
            ty = np.sin(angle + np.sign(cp) * np.pi / 2)

            dx[k] += tangent_ratio * kappa[k] * tx
            dy[k] += tangent_ratio * kappa[k] * ty

            dx[k] += normal_ratio * kappa[k] * nx
            dy[k] += normal_ratio * kappa[k] * ny

        # backup length before deformation
        length_prev = cumulative_distance(x, y, normalized=False)[-1]

        # apply deformation
        dx = smoothing_gaussian(dx, ir)
        dy = smoothing_gaussian(dy, ir)

        x += dx * shape_factor
        y += dy * shape_factor

        # reinterpolate to get a fairly constant discretization size
        length = cumulative_distance(x, y, normalized=False)[-1]
        new_npoints = min(
            3 * npoints_mini,
            max(npoints_mini, int(x.size * length / length_prev)))
        x, y = interp_spline(x, y, npoints=new_npoints)

        # remove oxbows
        x, y, loops_x_it, loops_y_it = remove_loops(x, y)

        loops_x += loops_x_it
        loops_y += loops_y_it

    return x, y, loops_x, loops_y


def plot_river(x, y, loops_x, loops_y):
    plt.figure()

    color = (0.2, 0.2, 0.2)
    for x_, y_ in zip(loops_x, loops_y):
        plt.plot(x_, y_, '-', lw=0.3, color=color)

    plt.plot(x, y, 'w-', lw=0.7)

    plt.xlim([-0.2, 1.2])
    plt.ylim([-0.2, 1.2])
    plt.gca().set_aspect('equal', 'box')
    plt.axis('off')
    return None


# remove loops of a curve
def remove_loops(x, y):
    x = np.copy(x)
    y = np.copy(y)

    loops_x = []
    loops_y = []

    k = 0
    while (k < x.size - 1):
        remove_node = False
        for p in range(k + 2, x.size - 1):
            a = (x[k], y[k])
            b = (x[k + 1], y[k + 1])
            c = (x[p], y[p])
            d = (x[p + 1], y[p + 1])
            if intersect(a, b, c, d):
                remove_node = True
                break

        if remove_node:
            loops_x.append(x[k + 1:p + 1])
            loops_y.append(y[k + 1:p + 1])

            x = np.delete(x, np.s_[k + 1:p + 1])
            y = np.delete(y, np.s_[k + 1:p + 1])
            k = p + 1
        else:
            k += 1

    return x, y, loops_x, loops_y


# apply laplacian smoothing
def smoothing_laplace(x, iterations=1):
    dx = np.zeros_like(x)
    for _ in range(iterations):
        dx[1:-1] = -0.25 * (x[:-2] + x[2:]) + 0.5 * x[1:-1]
        x = x - dx
    return x


def smoothing_gaussian(x, ir):
    kernel = cubic_pulse(2 * ir + 1)
    kernel = kernel / np.sum(kernel)
    x = scipy.ndimage.convolve1d(x, kernel, mode='reflect')
    return x


def tricube(npoints):
    r = np.abs(np.linspace(-1, 1, npoints))
    return (1 - r**3)**3
