# Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU
# General Public License. The full license is in the file LICENSE,
# distributed with this software.

import matplotlib.pyplot as plt
import numpy as np
import os
#
import tools

plt.style.use('dark_background')
os.makedirs('frames', exist_ok=True)

seed = 6
npt = 10
npoints_spline = 400

# --- first start with a set of random nodes

rng = np.random.default_rng(seed)

x = rng.random(npt)
y = rng.random(npt)

# --- try to make a continuous path with these nodes using a nearest
# --- neighbor search

x, y = tools.nearest_neighbor_search_2d(x, y)

tools.plot_river(x, y, [], [])
plt.savefig('frames/r_init.png', dpi=180, bbox_inches='tight')

# --- add some midpoint displacement to ignite some large scale
# --- meandering (this step is not mandatory but by use alone can
# --- produce a convinving meandering pattern)

xm, ym = tools.meander_midpoint(x, y, amp=0.2, iterations=2)

tools.plot_river(xm, ym, [], [])
plt.savefig('frames/r_midpoint.png', dpi=180, bbox_inches='tight')

# --- spline interpolation to produce a smooth starting point for the
# --- meandering

xs, ys = tools.interp_spline(xm, ym, npoints=npoints_spline)

tools.plot_river(xs, ys, [], [])
plt.savefig('frames/r_spline.png', dpi=180, bbox_inches='tight')

# --- perform the meandering

# where all the dead loops (oxbows) are stored
loops_x = []
loops_y = []

iterations = 100

# Gaussian filtering radius (a lot of filtering is requested to avoid
# number instabilities, and the filter radius will tend to drive the
# lengthscale of the meanders)
ir = 5

# use a uniform normalization coefficient
curvature_normalization = 1 / 0.001  # inverse of reference radius

sub_iterations = 1

for it in range(iterations):
    print('iteration: ', it)

    xs, ys, loops_x_it, loops_y_it = tools.meander_strengthen(
        xs,
        ys,
        iterations=sub_iterations,
        ir=ir,
        tangent_ratio=0.02,
        normal_ratio=0.01,
        npoints_mini=npoints_spline,
        curvature_normalization=curvature_normalization)

    # uncomment to plot the loops (memory consuming)
    # loops_x += loops_x_it
    # loops_y += loops_y_it

    tools.plot_river(xs, ys, loops_x, loops_y)

    plt.savefig('frames/river_' + str(it).zfill(4) + '.png',
                dpi=180,
                bbox_inches='tight')
    plt.close()

tools.plot_river(xs, ys, [], [])
plt.savefig('frames/r_final.png', dpi=180, bbox_inches='tight')

plt.show()
