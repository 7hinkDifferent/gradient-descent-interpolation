'''
interpolate value given existing data points
'''

# TODO: a more efficient implementation
class LagrangePolynomialInterpolation(object):
    def __call__(self, x, x_values, y_values):
        try:
            n = len(x_values)
        except TypeError:
            raise TypeError("x_values should be iterable")

        result = 0.0
        # O(n^2)
        for i in range(n):
            term = y_values[i]
            for j in range(n):
                if j != i:
                    term = term * (x - x_values[j]) / (x_values[i] - x_values[j])
            result += term
        return result
    
