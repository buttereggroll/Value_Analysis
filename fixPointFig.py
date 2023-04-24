# # -*- coding: utf-8 -*-
# """
# Created on Thu Apr 21 20:31:34 2022
#
# @author: amaze
# """

from matplotlib import pyplot as plt
import numpy as np
from matplotlib import animation


def fixpt(f, x, epsilon=1.0E-5, N=500, store=False):
    y = f(x)
    n = 0
    Values = []
    if store:
        Values.append((x, y))
    while abs(y-x) >= epsilon and n < N:
        x = f(x)
        n += 1
        y = f(x)
        if store:
            Values.append((x, y))
    if store:
        return y, Values
    else:
        if n >= N:
            return "No fixed point for given start value"
        else:
            return x, n, y


# define f
def f(x):
     return ((-58 * x - 3) / (7 * x ** 3 - 13 * x ** 2 - 21 * x - 12)) ** (1 / 2)


# find fixed point
res, points = fixpt(f, 1.5, store=True)

# # create mesh for plots
# xx = np.arange(1.2, 1.6, 1e-5)

# i = list(range(1, 10))


# def update(i):
#     """在每个时间点操作图形对象"""
#     plt.clf()  # 清空图层
#
#     x, y = points[i][0], points[i][1]
#     distance = abs(y - x)
#     plt_x1 = np.linspace(start=y - distance, stop=y + distance, num=300)
#     plt_y1 = f(plt_x1)
#     plt_x2 = plt_x1
#     plt_y2 = plt_x2
#     plt.title('{}'.format(i))
#     plt.plot(plt_x1, plt_y1, 'b')
#     plt.plot(plt_x2, plt_y2, 'r')
#     plt.plot([x, x], [x, y], 'g')
#     plt.plot([x, y], [y, y], 'g')
#
#
# fig = plt.figure()
# ani = animation.FuncAnimation(fig, update)
#
# plt.show()
# plot function and identity
def func_id(xx):
    plt.plot(xx, f(xx), 'b')
    plt.plot(xx, xx, 'r')

iter_num = 1
d = 1.5 - res
# plot lines
for x, y in points:
    # distance = abs(x - y)
    # xx = np.arange(1.308925765 - distance, 1.308925765 + 2 * distance, 1e-6)
    # xx = np.arange(y - distance, y + distance, 1e-6)
    # if iter_num <= 6:
    #     xx = np.arange(1.3, 1.5, 1e-6)
    # elif iter_num <= 10:
    #     xx = np.arange(1.308, 1.312, 1e-6)
    # elif iter_num <= 13:
    #     xx = np.arange(1.3089, 1.309, 1e-6)
    # else:
    #     xx = np.arange(1.30892, 1.30893, 1e-6)
    if (x - y) / d <= 1 / 20:
        d = x - res
    xx = np.arange(res, res + d, 1e-6)
    func_id(xx)
    plt.plot([x, x], [x, y], 'g')
    plt.title(iter_num)
    iter_num += 1
    plt.pause(0.1)
    func_id(xx)
    plt.plot([x, y], [y, y], 'g')
    plt.title(iter_num)
    iter_num += 1
    plt.pause(0.1)

# show result
plt.show()
