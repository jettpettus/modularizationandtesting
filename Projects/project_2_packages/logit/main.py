import numpy as np
from math import log, exp
from scipy.optimize import minimize


def ll_function (x, y, beta):
    # Reshape beta

    ll = 0
    for i in range(1, x.shape[0]):
        ll = ll + exp(np.dot(x[i, :], beta)) + y[i] * np.dot(x[i, :], beta)
    return ll

def fit_logit(x, y):
    beta_init = np.random.rand(x.shape[1], 1)
    min_func = (lambda beta: ll_function(x, y, beta))
    res = minimize(min_func, x0 = beta_init)
    return res

def logit_se(fit):
    hessian_inv = fit.hess_inv
    se = np.sqrt(np.diag(hessian_inv))
    return se

if __name__ == "__main__":
    print("x")
    x = np.transpose(np.matrix([[0, 1, 3, 0.5, 3, -1, 2],
                               [1, 3, .2, 1, 4, 4, 1 ],
                               [2, 4, 9, 1, 0.2, 1, .4]]))
    print("dot")
    print(np.dot(x[1, :],  [1, 2, 3]))
    print(x)
    print(x.shape)
    print("y")
    y = np.transpose(np.matrix([9, 4, 23, 4, 4, -2, 1]))
    print("llfunct")
    ll_function(x, y, beta = [1, 2, 3])
    print(y)
    print(y.shape)
    fit = fit_logit(x, y)
    print(fit)
    print("se")
    print(logit_se(fit))
    # print(fit_logit(x, y))
