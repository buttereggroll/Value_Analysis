from matplotlib import pyplot as plt
from solve_linear_systems import *


def Lg(data, testdata):
    predict = 0
    data_x = [data[i][0] for i in range(len(data))]
    data_y = [data[i][1] for i in range(len(data))]
    if testdata in data_x:
        # print " testdata is already known "
        return data_y[data_x.index(testdata)]
    for i in range(len(data_x)):
        af = 1
        for j in range(len(data_x)):
            if j != i:
                af *= (1.0 * (testdata - data_x[j]) / (data_x[i] - data_x[j]))
        predict += data_y[i] * af
    return predict


def Lg_plot(data, nums):
    data_x = [data[i][0] for i in range(len(data))]
    data_y = [data[i][1] for i in range(len(data))]
    Area = [min(data_x), max(data_x)]
    X = [Area[0] + 1.0 * i * (Area[1] - Area[0]) / nums for i in range(nums)]
    X[len(X) - 1] = Area[1]
    Y = [Lg(data, x) for x in X]
    plt.plot(X, Y, label='result')
    for i in range(len(data_x)):
        plt.plot(data_x[i], data_y[i], 'ro', label="point")
    plt.title('lg')
    plt.savefig('Lg.jpg')
    plt.show()


def calF(data):
    # 差商计算 n 个数据0-( n -1）阶个差商 n 个数据
    data_x = [data[i][0] for i in range(len(data))]
    data_y = [data[i][1] for i in range(len(data))]
    # F = np.ones(len(data))  # [1 for i in range(len(data))]
    FM = []
    for i in range(len(data)):
        FME = []
        if i == 0:
            FME = data_y
        else:
            for j in range(len(FM[len(FM) - 1]) - 1):
                delta = data_x[i + j] - data_x[j]
                value = 1.0 * (FM[len(FM) - 1][j + 1] - FM[len(FM) - 1][j]) / delta
                FME.append(value)
        FM.append(FME)
    F = [fme[0] for fme in FM]
    # print(FM)
    return F


def NT(data, testdata):
    # 差商之类的计算
    F = calF(data)
    predict = 0
    data_x = [data[i][0] for i in range(len(data))]
    data_y = [data[i][1] for i in range(len(data))]
    if testdata in data_x:
        return data_y[data_x.index(testdata)]
    else:
        for i in range(len(data_x)):
            Eq = 1
            if i != 0:
                for j in range(i):
                    Eq = Eq * (testdata - data_x[j])
                predict += (F[i] * Eq)
    return predict


def NT_plot(data, nums):
    data_x = [data[i][0] for i in range(len(data))]
    data_y = [data[i][1] for i in range(len(data))]

    Area = [min(data_x), max(data_x)]

    X = [Area[0] + 1.0 * i * (Area[1] - Area[0]) / nums for i in range(nums)]
    X[len(X) - 1] = Area[1]

    Y = [NT(data, x) for x in X]  # 牛顿插值

    plt.plot(X, Y, label='result')
    for i in range(len(data_x)):
        plt.plot(data_x[i], data_y[i], 'ro', label="point")
    plt.title('Newton')
    plt.savefig('Newton.jpg')
    plt.show()


def DivideLine(data, testdata):
    # 找到最邻近的
    data_x = [data[i][0] for i in range(len(data))]
    data_y = [data[i][1] for i in range(len(data))]

    if testdata in data_x:
        return data_y[data_x.index(testdata)]
    else:
        index = 0
        for j in range(len(data_x) - 1):
            if data_x[j] < testdata < data_x[j + 1]:
                index = j
                break
        predict = 1.0 * (testdata - data_x[index]) * (data_y[index + 1] - data_y[index]) / (
                data_x[index + 1] - data_x[index]) + data_y[index]
        return predict


def Divline_plot(data, nums):
    data_x = [data[i][0] for i in range(len(data))]
    data_y = [data[i][1] for i in range(len(data))]

    Area = [min(data_x), max(data_x)]

    X = [Area[0] + 1.0 * i * (Area[1] - Area[0]) / nums for i in range(nums + 1)]

    Y = [DivideLine(data, x) for x in X]

    plt.plot(X, Y, label='result')
    for i in range(len(data_x)):
        plt.plot(data_x[i], data_y[i], 'ro', label="point")
    plt.title('DivLine')
    plt.savefig('DivLine.jpg')
    plt.show()


def get_h(data):
    """输入列表data，得到列表h"""
    h = []
    for i in range(len(data) - 1):
        h.append(data[i + 1][0] - data[i][0])
    return h


def get_d(data):
    """输入列表data，得到列表d"""
    h = get_h(data)
    d = []
    for i in range(len(data) - 1):
        d.append((data[i + 1][1] - data[i][1]) / h[i])
    return d


def get_u(data):
    """输入列表d，得到列表u"""
    d = get_d(data)
    u = [0]
    for i in range(1, len(d)):
        u.append(6 * (d[i] - d[i - 1]))
    return u


def get_m(data, m0=0, mn=0):
    """tridiagonal_matrix_algorithm,输入h的列表，u的列表和[m0,mn](默认[0,0])，输出得到的解集列表m(第1到第n-1个m)"""
    h = get_h(data)
    u = get_u(data)
    size = len(h) - 1
    tri = np.zeros((size, size))
    b = np.zeros(size).T

    if size < 1:
        return [m0, mn]

    for i in range(size):
        if i > 0:
            tri[i - 1] = h[i]
        tri[i] = 2 * (h[i] + h[i + 1])
        if i < size - 1:
            tri[i + 1] = h[i + 1]

    for i in range(size):
        if i == 0:
            b[0] = u[1] - h[0] * m0
        elif i == size - 1:
            b[i] = u[size - 1] - h[size - 1] * mn
        else:
            b[i] = u[i + 1]

    solve = list(gauss_seidel(get_merge_matrix(tri, b)))
    solve.insert(0, m0)
    solve.append(mn)
    m = solve

    return m


def get_s(data):
    """输入列表m,d,h,data,输出列表s"""
    m = get_m(data)
    d = get_d(data)
    h = get_h(data)
    interval_num = len(h)
    s = []

    for i in range(interval_num):
        si1 = d[i] - h[i] * (2 * m[i] + m[i + 1]) / 6
        si2 = m[i] / 2
        si3 = (m[i + 1] - m[i]) / (6 * h[i])
        si = [data[i][1], si1, si2, si3]
        s.append(si)

    return s


def spline(data, testdata):
    """输入列表s,data，输出在对应区间的三次样条函数得到的函数值"""
    s = get_s(data)
    result = 0
    interval_num = len(data) - 1
    index = interval_num - 1

    for i in range(interval_num):
        if data[i][0] <= testdata < data[i+1][0]:
            index = i
            break

    for i in range(4):
        result += s[index][i] * (testdata - data[index][0]) ** i

    return result


def spline_plot(data, nums):
    """输出样条插值法构造函数的图像"""
    data_x = [data[i][0] for i in range(len(data))]
    data_y = [data[i][1] for i in range(len(data))]

    Area = [min(data_x), max(data_x)]

    X = [Area[0] + 1.0 * i * (Area[1] - Area[0]) / nums for i in range(nums + 1)]

    Y = [spline(data, x) for x in X]

    plt.plot(X, Y, label='result')
    for i in range(len(data_x)):
        plt.plot(data_x[i], data_y[i], 'ro', label="point")
    plt.title('Spline')
    plt.savefig('Spline.jpg')
    plt.show()
