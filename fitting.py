import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.optimize import leastsq

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号‘-’显示为方块的问题

x_observed = np.arange(1994, 2004, 1, dtype='float')  # 给定拟合的x值
x_predict = np.arange(2003, 2011, 1, dtype='float')  # 给定参与预测的x值
y_observed = np.array([67.052, 68.008, 69.803, 72.024, 73.400, 72.063,
                       74.669, 74.487, 74.065, 76.777])  # 给定拟合的y值

# 作图
plt.figure()
plt.title(u'石油产量变化')
plt.xlabel(u'年')
plt.ylabel(u'桶/年')
# 坐标轴的范围x_min, x_max, y_min, y_max
# plt.axis([1993.5, 2003.5, 66, 77])
plt.grid(True)
# plt.plot(x_observed, y_observed, 'k.')


# 直线函数
param0 = np.zeros(2)


def linear_fun(s, x):
    k, b = s
    return k * x + b


# 抛物线函数
param1 = np.zeros(3)


def quadratic_fun(s, x):
    k1, k2, b = s
    return k1 * x ** 2 + k2 * x + b


# 立方曲线
param2 = np.zeros(4)


def cubic_fun(s, x):
    k1, k2, k3, b = s
    return k1 * x ** 3 + k2 * x ** 2 + k3 * x + b


# 求出残差
def dist(a, fun, x, y):
    return fun(a, x) - y


funcs = [linear_fun, quadratic_fun, cubic_fun]
params = [param0, param1, param2]
colors = ['blue', 'green', 'red']
fun_names = ['linear_fun', 'quadratic_fun', 'cubic_fun']

for i, (func, param, color, name) in enumerate(zip(funcs, params,
                                                   colors, fun_names)):
    var = leastsq(dist, param, args=(func, x_observed, y_observed))
    # plt.plot(x_observed, func(var[0], x_observed), color)
    plt.plot(x_predict, func(var[0], x_predict), color)
    print('[%s] 二范数的平方: %.4f, abs(bias): %.4f, bias-std: %.4f' %
          (name, ((y_observed - func(var[0], x_observed)) ** 2).sum(),
           (y_observed - func(var[0], x_observed)).std(),
           (abs(y_observed - func(var[0], x_observed))).mean()
           ))
    print(var[0])
    print(name, '预测的2010年石油产量为：', func(var[0], 2010), '\n')

plt.legend(['sample data', 'linear_fun', 'quadratic_fun',
            'cubic_fun'], loc='upper left')
# plt.legend(['linear_fun', 'quadratic_fun',
#             'cubic_fun'], loc='upper left')
plt.show()
