import numpy as np

from square_root_method import generate_matrices_to_solve, calculate_conflict_vector

def calculate_x_jacobi(A, b, tolerance=1e-6, max_iter=1000):
    """
    Calculates the solution of a matrix equation Ax=b using Jacobi's Iterative Method 
    Input: A - square matrix of size (n, n),
    b - vector of size (n, ),
    tolerance - number that sets the calculations precision,
    max_iter - number that sets the maximum amount of iterations in the function call
    Output: x - vector of size (, n), the solution of the equation,
    k - number, the amount of iterations run until achieving set precision
    """
    n = A.shape[0]
    x = np.full(n, 0.5)

    for i in range(n):
        diag = A[i, i]
        diff = np.sum(np.abs(A[i, :])) - np.abs(diag)
        if diag <= diff:
            A[i, i] = diff + 1

    for k in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(A.shape[0]):
            s = np.dot(A[i, :], x) - A[i, i] * x[i]
            x_new[i] = (b[i] - s) / A[i, i]
   
        if np.max(np.abs(x_new - x)) < tolerance:
            break
        x = x_new

    return x, k


if __name__ == '__main__':
    A, b = generate_matrices_to_solve(7)
    x, k = calculate_x_jacobi(A, b)
    r = calculate_conflict_vector(A, b, x)

    print(f'solution x: {x}\n')
    print(f'conflict vector: {r}\n')
    print(f'amount of iterations: {k}')

