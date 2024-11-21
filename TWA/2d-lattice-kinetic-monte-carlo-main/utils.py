import numpy as np


def find_closest_perfect_square(x: int):

    sqrt_x = np.floor(np.sqrt(x))

    x_1 = sqrt_x ** 2
    x_2 = (sqrt_x + 1) ** 2

    dx_1 = x - x_1
    dx_2 = x_2 - x

    if dx_2 >= dx_1:
        print(f'The closest perfect square for x={x} is x_2={int(x_2)}.')
    else:
        print(f'The closest perfect square for x={x} is x_1={int(x_1)}.')


if __name__ == '__main__':
    find_closest_perfect_square(2500)
