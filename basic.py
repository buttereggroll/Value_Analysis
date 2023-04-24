import numpy as np


# 做题用的函数f1到f5
def f1(x):
    return 7.0 * x ** 5 - 13 * x ** 4 - 21 * x ** 3 - 12 * x ** 2 + 59 * x + 3


# 调用numpy包进行运算以防止出现复数
def f2(x):
    return np.power(((13.0 * x ** 4 + 21 * x ** 3 + 12 * x ** 2 - 58 * x - 3) / 7), (1 / 5))


def f3(x):
    return (13.0 + 21 / x + 12 / (x ** 2) - 58 / (x ** 3) - 3 / (x ** 4)) / 7


def f4(x):
    return np.power((12.0 * x ** 2 - 58 * x - 3) / (7 * x ** 2 - 13 * x - 21), 1 / 3)


def f5(x):
    return np.power((-58.0 * x - 3) / (7 * x ** 3 - 13 * x ** 2 - 21 * x - 12), 0.5)


def f(x):
    return 7.0 * x ** 5 - 13 * x ** 4 - 21 * x ** 3 - 12 * x ** 2 + 58 * x + 3


def f_derivative(x):
    return 35.0 * x ** 4 - 52 * x ** 3 - 63 * x ** 2 - 24 * x + 58


def aitken_iter(x0, x1, x2):
    return x0 - (x1 - x0) ** 2 / (x2 - 2 * x1 + x0)


def steffensen_iter(x0, function):
    return x0 - (function(x0) - x0) ** 2 / (function(function(x0)) - 2 * function(x0) + x0)


def newton_iter(x, function, derivative_function):
    return x - function(x) / derivative_function(x)


def static_secant_iter(x, x0, function):
    return x - (x - x0) * function(x) / (function(x) - function(x0))


def dynamic_secant_iter(x, pre_x, function):
    return x - (x - pre_x) * function(x) / (function(x) - function(pre_x))


def elimination(matrix):
    """输入一个numpy.array类型的矩阵，对矩阵代替的方程消元，返回一个上三角矩阵"""
    for i in range(len(matrix[0, :]) - 2):
        for j in range(i + 1, len(matrix[:, 0])):
            temp = matrix[j, i] / matrix[i, i]
            matrix[j, :] -= matrix[i, :] * temp
    return matrix


def back_substitution(matrix):
    """输入一个numpy.array类型的矩阵"""
    result = {}
    for i in range(len(matrix[:, 0]), 0, -1):
        temp = 0
        for j in range(len(matrix[:, 0]), i, -1):
            temp += result[j] * matrix[i - 1, j - 1]
        result[i] = (matrix[i - 1, len(matrix[0, :]) - 1] - temp) / matrix[i - 1, i - 1]
    return result


def maximal_column_pivoting(matrix, index):
    """输入一个numpy.array类型的矩阵，一个起始行主元序号index，按列选出绝对值最大的元素作主元"""
    max_num = matrix[index, index]
    pivot = index
    for i in range(index + 1, len(matrix[:, 0])):
        if max_num < abs(matrix[i, index]):
            pivot = i
            max_num = abs(matrix[i, index])
    if pivot != index:
        temp = matrix[pivot, :].copy()
        matrix[pivot, :] = matrix[index, :]
        matrix[index, :] = temp


def elimination_partial_pivoting(matrix):
    """输入一个numpy.array类型的矩阵，对矩阵代替的方程消元"""
    for i in range(len(matrix[:, 0]) - 1):
        maximal_column_pivoting(matrix, i)
        for j in range(i + 1, len(matrix[:, 0])):
            temp = float('{:.3f}'.format(matrix[i, i] / matrix[j, i]))
            matrix[j, :] -= matrix[i, :] / temp


def elimination_gauss_jordan(matrix):
    """假定输入的矩阵代表的方程组有唯一解，输入一个高斯消元后的矩阵，继续对它进行高斯-若当消元"""
    for i in range(len(matrix[:, 0])-1, -1, -1):
        matrix[i, :] /= matrix[i, i]
        for j in range(i-1, -1, -1):
            matrix[j, :] -= matrix[i, :] * matrix[j, i]


def get_lower_triangular_matrix(source_matrix):
    """输入一个numpy.array类型的代表线性方程组的矩阵，分解该矩阵A=LU，该方法返回其中的下三角矩阵L"""
    lower_triangular_matrix = np.zeros((len(source_matrix[:, 0]), len(source_matrix[0, :]) - 1))
    for i in range(len(source_matrix[:, 0])):
        for j in range(i, len(source_matrix[:, 0])):
            lower_triangular_matrix[j, i] = source_matrix[j, i] / source_matrix[i, i]
    return lower_triangular_matrix


def get_b(matrix):
    """输入一个代表线性方程组的矩阵，返回一个numpy.array()类型的由它等号右边组成的列向量"""
    temp_list = []
    for i in range(len(matrix[:, 0])):
        temp_list.append(matrix[i, len(matrix[0, :]) - 1])
    return np.array(temp_list)


def get_merge_matrix(matrix, b):
    """输入一个矩阵和一个列向量b，返回它们左右合并后的矩阵"""
    temp_matrix = np.zeros((len(matrix[:, 0]), len(matrix[0, :]) + 1))
    for i in range(len(temp_matrix[:, 0])):
        temp_matrix[i, len(temp_matrix[0, :]) - 1] = b[i]
        for j in range(len(matrix[:, 0])):
            temp_matrix[j, i] = matrix[j, i]
    return temp_matrix


def get_intermediate_y(lower_triangular_matrix, b):
    """输入一个下三角矩阵和一个列向量b(相当于线性方程组等号的右边)，令LUx=b中的Ux=y，求解Ly=b，以numpy.array()的形式返回y"""
    temp_matrix = get_merge_matrix(lower_triangular_matrix, b)
    y = []
    for i in range(len(temp_matrix[:, 0])):
        temp = 0
        for j in range(0, i):
            temp += y[j] * temp_matrix[i, j]
        y.append(temp_matrix[i, len(temp_matrix[0, :]) - 1] - temp)
    return np.array(y)


def get_iter_matrix(matrix):
    """输入numpy.array()类型的由系数矩阵和方程组右端列向量合成的矩阵，返回迭代系数矩阵和迭代常数列向量"""
    iter_matrix = np.zeros((len(matrix[:, 0]), len(matrix[0, :]) - 1))
    iter_array = []
    for i in range(len(matrix[:, 0])):
        iter_array.append(matrix[i, len(matrix[0, :]) - 1] / matrix[i, i])
    i = 0
    while i < len(matrix[:, 0]):
        j = 0
        while j < len(matrix[0, :]) - 1:
            if j is not i:
                iter_matrix[i, j] = -1 * matrix[i, j] / matrix[i, i]
            else:
                iter_matrix[i, i] = 0
            j += 1
        i += 1
    return iter_matrix, np.transpose(iter_array)


def jacobi_iter(iter_matrix, iter_array, x_list):
    """利用Jacobi迭代式计算每一轮的迭代结果
        输入numpy.array()类型的迭代系数矩阵，迭代常数向量和第i轮解集
        输出numpy.array()类型的第i+1轮的解集"""
    return np.dot(iter_matrix, x_list) + iter_array


def gauss_seidel_iter(iter_matrix, iter_array, x_list):
    """利用高斯赛德尔迭代式计算每一轮的迭代结果
    输入numpy.array()类型的迭代矩阵，迭代常数向量和第i轮解集
    输出numpy.array()类型的第i+1轮的解集"""
    temp_list = x_list.copy()
    for i in range(len(temp_list)):
        temp_list[i] = np.dot(iter_matrix[i, :], temp_list) + iter_array[i]
    return temp_list


def spectral_radius(matrix):
    """计算谱半径"""
    eigenvalue_list = np.linalg.eigvals(matrix)
    max_spectral_radius = 0
    for eigenvalue in eigenvalue_list:
        if abs(eigenvalue) > max_spectral_radius:
            max_spectral_radius = abs(eigenvalue)
    return max_spectral_radius


def get_hilbert(n):
    """生成n阶希尔伯特矩阵"""
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = 1 / (i + j + 1)
    return matrix


def derive_b(matrix, x):
    """由已知的解和系数矩阵推导出方程右端列向量b"""
    column_num = len(matrix[:, 0])
    b = np.zeros((column_num, 1))
    for i in range(column_num):
        b[i] = np.dot(matrix[i, :], x)
    return b
