import numpy as np
import numpy.typing as npt
from math import fabs
from random import uniform

EPS, INF = 1e-8, 1e9
num_iterations = 10_000

def random_real_matrix_from_real_eigenvals(eigvals: npt.NDArray[np.float64], n: int) -> npt.NDArray[np.float64]:
    '''
    T^(-1) * D * T
    O(n^3)
    Parameters:
        eigvals: (n) ndarray
        n : int
    Returns:
        Matrix with spectrum = eigenvals
    '''
    T = np.random.random((n,n))
    return np.linalg.inv(T)@np.diag(eigvals)@T

def sign(x):
    return 1 if (x >= -EPS) else -1

def Hessenberg(A: npt.NDArray[np.float64], n: int) -> npt.NDArray[np.float64]:
    for k in range(1,n-1):
        s_k = sign(A[k,k-1]) * sum(A[i,k-1]*A[i,k-1] for i in range(k,n))**0.5
        mu_k = 1/(2*s_k*(s_k - A[k,k-1]))**0.5
        v = np.array([0.0]*n)
        v[k] = A[k,k-1] - s_k
        for i in range(k+1,n): v[i] = A[i,k-1]
        v *= mu_k
        H_k = np.eye(n) - 2*v.reshape(n,1)@v.reshape(1,n)
        A = H_k@A@H_k
    return A

def find_eigenvals(A: npt.NDArray[np.float64], n: int) -> tuple[npt.NDArray[np.float64], int]:
    k = 0
    ans = []
    while (k < num_iterations):
        tmp = A[n-1,n-1]
        Q_k,R_k = np.linalg.qr(A - A[n-1,n-1]*np.eye(n))
        A = R_k@Q_k + A[n-1,n-1]*np.eye(n)
        k += 1
        if ((n == 1 or fabs(A[n-1,n-2]) < EPS) and fabs(A[n-1,n-1] - tmp) < fabs(tmp)/3):
            ans.append(A[n-1,n-1])
            A = A[:n-1,:n-1]
            n -= 1
            if (not n): break
    return np.array(ans),k

def inverse_iterations(A,n,sigma):
    y_k = np.random.rand(n)
    z_k = y_k/np.linalg.norm(y_k)
    mu_k = np.array([INF]*n)
    
    z_prev, mu_prev = z_k.copy(), mu_k.copy()
    k = 1
    while (k < num_iterations):
        flag = True
        y_k = np.linalg.solve(A - sigma*np.eye(n),z_prev)
        z_k = y_k/np.linalg.norm(y_k)
        for i in range(n): 
            
            mu_k[i] = INF if (fabs(y_k[i]) < EPS) else z_prev[i]/y_k[i]
            if (fabs(mu_k[i] - mu_prev[i]) > EPS): flag = False
        
        if (flag): break
        z_prev[:], mu_prev[:] = z_k, mu_k
        
        k += 1
    return mu_k.sum()/n + sigma, z_k, k

def main() -> int:
    global l, eigvals, A
    n = 4
    eigvals = np.random.rand(n)
    A = random_real_matrix_from_real_eigenvals(eigvals,n)
    l,k = find_eigenvals(Hessenberg(A,n),n)
    print("approximate max|λ| =",max(abs(l)))
    print("exact max|λ| =",max(abs(eigvals)))
    print("error =",fabs(max(abs(l)) - max(abs(eigvals))))
    print("in %d iterations"%k)
    print()
    new_l = []
    new_k = 0
    for sigma in l:
        l,z,k = inverse_iterations(A,n,sigma)
        new_l.append(l)
        new_k += 1
    new_l = np.array(new_l)
    print("approximate max|λ| =",max(abs(new_l)))
    print("exact max|λ| =",max(abs(eigvals)))
    print("error =",fabs(max(abs(new_l)) - max(abs(eigvals))))
    print("in %d iterations"%new_k)
    return 0

if (__name__ == "__main__"):
    main()