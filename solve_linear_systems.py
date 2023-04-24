from basic import *


def gauss_method(matrix):
    """高斯消元法，输入一个numpy.array类型的矩阵，返回一个字典（字典序号表示第i个解，i = 1, 2, 3...）"""
    elimination(matrix)
    return back_substitution(matrix)


def gauss_column_pivoting(matrix):
    """高斯列主元消元法，输入一个numpy.array类型的矩阵，返回一个字典（字典序号表示第i个解，i = 1, 2, 3...）"""
    elimination_partial_pivoting(matrix)
    return back_substitution(matrix)


def gauss_jordan_method(matrix):
    """高斯-若当消元法，输入一个numpy.array类型的矩阵，返回一个字典（字典序号表示第i个解，i = 1, 2, 3...）"""
    elimination(matrix)
    elimination_gauss_jordan(matrix)
    dic = {}
    for i in range(len(matrix[:, 0])):
        dic[i+1] = matrix[i, len(matrix[0, :]) - 1]
    return dic


def gauss_jordan_list(matrix):
    """高斯-若当消元法，输入一个numpy.array类型的矩阵，返回一个列表"""
    elimination(matrix)
    elimination_gauss_jordan(matrix)
    lis = []
    for i in range(len(matrix[:, 0])):
        lis.append(matrix[i, len(matrix[0, :]) - 1])
    return lis


def doolittle_method(matrix):
    """输入一个矩阵，求解LUx=b(相当于求解线性方程组Ax=b)，返回一个字典（字典序号表示第i个解，i = 1, 2, 3...）"""
    upper_triangular_matrix = elimination(matrix)       # 得到一个上三角矩阵
    lower_triangular_matrix = get_lower_triangular_matrix(matrix)
    b = get_b(matrix)
    y = get_intermediate_y(lower_triangular_matrix, b)
    return back_substitution(get_merge_matrix(upper_triangular_matrix, y))


def matrix_norm_1(matrix):
    """计算矩阵1范数（列和范数），输入一个numpy.array()类型的方阵(其实也就是numpy.ndarray)，返回它的标量值"""
    max = 0
    for i in range(len(matrix[0, :])):
        temp = 0
        for j in range(len(matrix[:, 0])):
            temp += abs(matrix[j, i])
        if max < temp:
            max = temp
    return max


def matrix_norm_infinite(matrix):
    """计算矩阵无穷范数（行和范数），输入一个numpy.array()类型的方阵(其实也就是numpy.ndarray)，返回它的标量值"""
    max = 0
    for i in range(len(matrix[:, 0])):
        temp = 0
        for j in range(len(matrix[0, :])):
            temp += abs(matrix[i, j])
        if max < temp:
            max = temp
    return max


def matrix_norm_spectral(matrix):
    """计算矩阵2范数（谱范数），输入一个numpy.array()类型的方阵(其实也就是numpy.ndarray)，返回它的标量值"""
    max_spectral_radius = spectral_radius(np.matmul(np.transpose(matrix), matrix))
    return np.sqrt(max_spectral_radius)


def cond(matrix):
    """计算矩阵的条件数，输入一个numpy.array()类型的方阵(其实也就是numpy.ndarray)，返回它的标量值"""
    return matrix_norm_spectral(np.linalg.inv(matrix)) * matrix_norm_spectral(matrix)


def relative_error(array_precise, array_perturb, order):
    """计算相对误差，输入两个numpy.array()类型的向量和指定范数，输出一个标量"""
    return np.linalg.norm(array_perturb - array_precise, order) / np.linalg.norm(array_precise, order)


def jacobi_method(matrix, x_init=None, limit=1e-6, max_iters=20):
    """
    解线性方程组的简单迭代法，输入numpy.array()类型的由系数矩阵和方程组右端列向量合成的矩阵
    和一个列表类型的初始解（默认全零）,返回一部字典（字典序号表示第i个解，i = 1, 2, 3...）
    默认最多迭代20次
    """
    x = []
    if x_init is None:
        x.append(np.zeros(len(matrix[:, 0])))
    else:
        x.append(x_init)

    # print('\njacobi迭代法:\n')
    # print("迭代次数：0, 当前解集：{}".format(x[-1]))
    iter_matrix, iter_array = get_iter_matrix(matrix)
    iter_times = 1
    x.append(jacobi_iter(iter_matrix, iter_array, x[-1]))
    while np.linalg.norm(x[-1] - x[-2], np.inf) >= limit and iter_times <= max_iters:
        # print("迭代次数：{}, 当前解集：{}".format(iter_times, x[-1]))
        x.append(jacobi_iter(iter_matrix, iter_array, x[-1]))
        iter_times += 1
    return x[-1]


def gauss_seidel(matrix, x_init=None, limit=1e-6, max_iters=5):
    """
    解线性方程组的改进迭代法，输入numpy.array()类型的由系数矩阵和方程组右端列向量合成的矩阵
    和一个列表类型的初始解（默认全零）,返回一部字典（字典序号表示第i个解，i = 1, 2, 3...）
    默认最多迭代5次
    """
    x = []
    if x_init is None:
        x.append(np.zeros(len(matrix[:, 0])))
    else:
        x.append(x_init)

    # print('\ngauss_seidel迭代法:\n')
    # print("迭代次数：0, 当前解集：{}".format(x[-1]))
    iter_matrix, iter_array = get_iter_matrix(matrix)
    iter_times = 1
    x.append(gauss_seidel_iter(iter_matrix, iter_array, x[-1]))
    while np.linalg.norm(x[-1] - x[-2], np.inf) >= limit and iter_times <= max_iters:
        # print("迭代次数：{}, 当前解集：{}".format(iter_times, x[-1]))
        x.append(gauss_seidel_iter(iter_matrix, iter_array, x[-1]))
        iter_times += 1
    return x[-1]


def turn_to_list(x_dic):
    """把字典表示的解集x转换为用列表表示"""
    x_list = []
    for i in range(1, len(x_dic.keys()) + 1):
        x_list.append(x_dic[i])
    return x_list
