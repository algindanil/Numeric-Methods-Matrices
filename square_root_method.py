import numpy as np
import matplotlib.pyplot as plt

def generate_matrices_to_solve(n):
    """
    Input: n - scalar\n
    Output: A - matrix of size (n,n), b - vector of size (n, ).\n
    Generated from the Lab 2, Variant 12 condition
    """

    a = lambda k: 1 + (n / 4) + (2 / k)
    p = lambda k: 4 / k

    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i][j] = a(i+1)
            elif j == i - 1:
                A[i][j] = p(i)
            elif i == j - 1:
                A[i][j] = p(j)
    A[n - 1][0] = 2
    A[0][n - 1] = 2

    b =  np.arange(n + 2, 2, -1).reshape(n, )

    return A, b


def generate_matrices_D_S(A):
    """
    Input: A - matrix of size (n, n)\n
    Output: D - matrix of size (n, n) containing ± values on the main diagonal and zeros elsewhere,
    S - matrix of size (n, n) 
    """

    n = A.shape[0]
    S = np.zeros((n, n))
    D = np.zeros((n, n))

    for i in range(n):
        sum_for_ii = 0
        sum_for_s_ij = 0

        for p in range(i):
            sum_for_ii += S[p][i]**2 * D[p][p]
        difference = A[i][i] - sum_for_ii
        D[i][i] = 1 if difference >= 0 else -1
        S[i][i] = np.sqrt(np.abs(difference))
        if i != n-1:
            for j in range(i + 1, n, 1):
                for p in range(i):
                    sum_for_s_ij += S[p][i] * D[p][p] * S[p][j]
                S[i][j] = (A[i][j] - sum_for_s_ij) / (D[i][i] * S[i][i])

    return D, S


def solve_upper_triangular(A, b):
    """
    Solves a matrix equation Ax=b. Assumes A is an upper triangular matrix
    Input: A - matrix of size (n, n),
    b - vector of size (n, )
    Output: vector of size (, n)
    """
    
    n = len(b)
    x = np.zeros(n)
    
    for i in range(n-1, -1, -1):
        x[i] = b[i]
        for j in range(i+1, n):
            x[i] -= A[i][j] * x[j]
        x[i] /= A[i][i]
    
    return x


def solve_lower_triangular(A, b):
    """
    Solves a matrix equation Ax=b. Assumes A is a lower triangular matrix
    Input: A - matrix of size (n, n),
    b - vector of size (n, )
    Output: vector of size (, n)
    """

    n = len(b)
    x = np.zeros(n)
    
    for i in range(n):
        x[i] = b[i]
        for j in range(i):
            x[i] -= A[i][j] * x[j]
        x[i] /= A[i][i]
    
    return x


def calculate_conflict_vector(A, b, x):
    """
    Input: A - square matrix of size (n, n),
    b - vector of size (n, ),
    x - vector of size (, n)
    Output: vector of size (n, ) containing the calculating errors corresponding to each component of vector b
    """

    return A @ x - b


def calculate_inverse_matrix(A):
    """
    Calculates the inverse matrix using the Jodan-Gauss Elimination 
    Input: A - square matrix of size (n, n), det(A) must not equal 0
    Output: square matrix of size (n, n), the inverse matrix of matrix A
    """
    n = len(A)
    identity = np.identity(n)
    augmented = np.concatenate([A, identity], axis=1)
    for i in range(n):
        divisor = augmented[i][i]
        augmented[i] /= divisor

        for j in range(n):
            if i != j:
                multiplier = augmented[j][i]
                augmented[j] -= multiplier * augmented[i]

    return augmented[:, n:]
    

def calculate_condition_number(A):
    """
    Calculates the condition number of a matrix using the 2-norm
    Input: A - square matrix of size (n, n), det(A) must not equal 0
    Output: the condition number of matrix A
    """
    norm_A = np.linalg.norm(A, ord=2)
    norm_A_inv = np.linalg.norm(np.linalg.inv(A), ord=2)

    return norm_A / norm_A_inv


def calculate_determinant(A, S, D):
    """
    Calculates the determinant of the matrix using the Square Root Method
    Input: A - square matrix of size (n, n),
    D - square matrix of size (n, n) generated from matrix A, containing ± values on the main diagonal and zeros elsewhere,
    S - square matrix of size (n, n) generated form matrix A
    Output: the determinant of matrix A
    """
    n = A.shape[0]
    det = 1
    
    for i in range(n):
        det *= D[i][i]
        det *= S[i][i] ** 2

    return det


def visualize_matrix(matrix):
    """
    Visualizes the matrix using Matplotlib
    Input: matrix
    """
    _, ax = plt.subplots()
    ax.matshow(matrix, cmap='Blues')

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, '{:.3f}'.format(matrix[i, j]), ha='center', va='center')

    plt.show()
    


if __name__ == '__main__':
    A, b = generate_matrices_to_solve(7)

    if np.array_equal(A, A.transpose()):
        D, S = generate_matrices_D_S(A)
        S_T = S.transpose()
        S_T_multiply_by_D = np.matmul(S_T, D)
        y = solve_lower_triangular(S_T_multiply_by_D, b)
        x = solve_upper_triangular(S, y)

        det = calculate_determinant(A, S, D)
        r = calculate_conflict_vector(A, b, x)
        cond = calculate_condition_number(A)
        A_inv = calculate_inverse_matrix(A)

        print(f'solution x: {x}\n')
        print(f'conflict vector: {r}\n')
        print(f'condiiton number: {cond}\n')
        print(f'determinant: {det}\n')
        visualize_matrix(A_inv)
        visualize_matrix((A_inv @ A))

    else:
        print('The matrix is not symmetrical. Error')
