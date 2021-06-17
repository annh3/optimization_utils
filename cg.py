import numpy as np

def conjugate_gradient(A,b,tol):
    x = np.random.rand(b.shape[0]) # unif in [0,1)
    r = b - A @x    # calculate residual
    if np.linalg.norm(r) < tol:
        return x
    p = r
    k = 0
    
    while(True):
        alpha = np.dot(r,r) / float(p.T @ A @ p)
        x = x + alpha * p
        r_old = r
        r = r_old - alpha * A @ p
        if np.linalg.norm(r) < tol:
            return x
        Beta = r.T @ r / r_old.T @ r_old
        p = r + Beta * p
        k = k + 1
        if k > 10 * b.shape[0]:
            return x