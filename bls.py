import numpy as np

def line_search(f, x_c, c, search_dir, rho, grad_dir, tol=1e-3, max_iter=100):
    alpha = 1
    iter = 0
    while iter < max_iter and f(x_c + alpha * search_dir) > f(x_c) + (c * alpha * grad_dir.T @ search_dir) + tol:
        alpha = rho * alpha
        iter = iter + 1
        if iter % 20 == 0:
        	print("iter: ", iter)
        	print("lhs: ", f(x_c + alpha * search_dir))
        	print("rhs: ", f(x_c) + (c * alpha * grad_dir.T @ search_dir) + tol)
        	print("\n")
    return alpha, f(x_c + alpha * search_dir)
