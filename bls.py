import numpy as np

def line_search(f, x_c, search_dir, gamma, grad_dir, tol):
    t = 1
    f_c = f(x_c)
    new_f = f(x_c + search_dir)
    derphi = np.dot(grad_dir, search_dir)
    while new_f > f_c + t*derphi + tol:
        t = gamma * t
        new_f = f(x_c + t*search_dir)
    return t
