import unittest
from basic import *
from interpolation import *
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_something(self):
        data = np.array([[0, 0], [1, 1], [2, 16]])
        self.assertEqual(spline(data, 1.5), 7.1875)

    def test_spline(self):
        data = np.array([[0, 0], [1, 2], [2, 3], [3, 8], [4, 2], [5, 7], [6, 8]])
        delta_x = 1e-10
        # data = np.array([[0, 0], [1, 1], [2, 16]])
        h = get_h(data)
        d = get_d(data)
        m = get_m(data)
        u = get_u(data)
        s = get_s(data)
        print(s, m, u, d, h)
        for i in range(1, 7):
            first_diriv1 = (spline(data, i) - spline(data, i - delta_x)) / delta_x
            first_diriv2 = (spline(data, i + delta_x) - spline(data, i)) / delta_x
            first_diriv3 = (spline(data, i - delta_x) - spline(data, i - 2 * delta_x)) / delta_x
            first_diriv4 = (spline(data, i + 2 * delta_x) - spline(data, i + delta_x)) / delta_x
            second_diriv1 = (first_diriv1 - first_diriv2) / delta_x
            second_diriv2 = (first_diriv3 - first_diriv4) / delta_x
            print(first_diriv1, first_diriv2, first_diriv3, first_diriv4, second_diriv1, second_diriv2)
            self.assertEqual(h[i-1]*m[i-1]+2*(h[i-1]+h[i])*m[i]+h[i]*m[i+1], u[i])
            # self.assertEqual(data[i][1], spline(data, i))
            # self.assertAlmostEqual(first_diriv1, first_diriv2)
            # self.assertAlmostEqual(second_diriv1, second_diriv2)

    # def test_get_s(self):


if __name__ == '__main__':
    unittest.main()
