import numpy as np

def lambda_spike_RMT(n, p, l):
    """
    Returns the covariance spike model based on the number of samples n,
    dimension p, and the eigenvalue shifts l.
    """
    return (n - p * (l** -2)) / (n + p * (l** -1))

def phase_transition_threshold(n, p):
    """
    Returns the phase transition threshold for PCA based on n and p.
    """
    return np.sqrt(p / n)

def frobenius_error(U_est, U_true):
    """
    Compute the Frobenius error between the estimated eigenspace and the true eigenspace.
    """
    return np.linalg.norm(U_est @ U_est.T - U_true @ U_true.T, 'fro')**2/2

def pooled_pca(data_machines, r):
    """
    Perform pooled PCA by summing all K sample covariance matrices and performing PCA on the total.
    """
    K = len(data_machines)
    n = data_machines[0].shape[0]
    
    # Compute sample covariance matrices for each machine
    cov_matrices = [np.cov(X.T) for X in data_machines]
    
    # Sum all the covariance matrices and perform PCA
    cov_total = np.sum(cov_matrices, axis=0)
    eigvals, eigvecs = np.linalg.eigh(cov_total)
    
    # Sort eigenvalues and select top r eigenvectors
    sorted_indices = np.argsort(eigvals)[::-1]
    Up = eigvecs[:, sorted_indices[:r]]
    
    return Up

def distributed_pca(data_machines, r): #with eigenvalue information
    """
    Perform distributed PCA by calculating the leading eigenvectors for each machine and combining them.
    """
    K = len(data_machines)
    
    # Compute sample covariance matrices for each machine
    cov_matrices = [np.cov(X.T) for X in data_machines]
    K = len(cov_matrices)  # 计算协方差矩阵的数量
    U_combined = np.zeros_like(cov_matrices[0])  # 初始化 U_combined 为适当大小的零矩阵

    for Sigma_k in cov_matrices:
        eigvals, eigvecs = np.linalg.eigh(Sigma_k)
        sorted_indices = np.argsort(eigvals)[::-1]  # 按降序排列特征值的索引
        Uk = eigvecs[:, sorted_indices[:r]]  # 取前 r 个特征向量
        lam = eigvals[sorted_indices[:r]]   # 对应的前 r 个特征值，已经排序过了
        Lam = np.diag(lam)                  # 构造对角矩阵
    
        term = Uk @ (Lam @ Uk.T)            # 计算加权后的贡献矩阵
        U_combined += term
    
    # Average the combined matrix
    U_combined /= K
    
    # Perform eigen-decomposition on the averaged matrix
    eigvals_combined, eigvecs_combined = np.linalg.eigh(U_combined)
        # Sort eigenvalues in descending order and select the top r eigenvectors
    sorted_indices = np.argsort(eigvals_combined)[::-1]
    Ud = eigvecs_combined[:, sorted_indices[:r]]
    
    return Ud

def distributed_pca_power(data_machines, r,T=1):
    K = len(data_machines)
    
    # Compute sample covariance matrices for each machine
    cov_matrices = [np.cov(X.T) for X in data_machines]
    
    Ug = distributed_pca(data_machines, r)
    for t in range(T):
        G_matrices = [cov@Ug for cov in cov_matrices]
        # Sum all the covariance matrices and perform PCA
        G_total = np.sum(G_matrices, axis=0)/K
        eigvals, eigvecs = np.linalg.eigh(G_total@G_total.T)
    
        # Sort eigenvalues and select top r eigenvectors
        sorted_indices = np.argsort(eigvals)[::-1]
        Ug = eigvecs[:, sorted_indices[:r]]
    return Ug

def distributed_pca_power_sigma(data_machines, r,T=1,sigma=None):
    K = len(data_machines)
    
    # Compute sample covariance matrices for each machine
    cov_matrices = [np.cov(X.T) for X in data_machines]
    Ug = distributed_pca(data_machines, r)
    p = Ug.shape[0]
    for t in range(T):
        G_matrices = [cov@Ug for cov in cov_matrices]
        if sigma is None:
            variance_total = np.sum([np.trace(cov) for cov in cov_matrices])
            variance_spike = np.sum([np.trace(Ug.T@cov@Ug) for cov in cov_matrices])
            sigma = np.sqrt((variance_total-variance_spike)/(K*(p-r)))
            print(sigma)
        # Sum all the covariance matrices and perform PCA
        G_total = np.sum(G_matrices-sigma*Ug, axis=0)/K
        eigvals, eigvecs = np.linalg.eigh(G_total@G_total.T)
    
        # Sort eigenvalues and select top r eigenvectors
        sorted_indices = np.argsort(eigvals)[::-1]
        Ug = eigvecs[:, sorted_indices[:r]]
    return Ug


def generate_data(l, n, K, p):
    """
    Generates datasets for K machines with n samples each, based on the spiked covariance model.
    """
    # Create the true covariance matrix using the provided eigenvalue shifts l
    Sigma_true = np.diag([1 + li for li in l] + [1] * (p - len(l)))

    # Generate datasets for K machines
    data_machines = []
    for _ in range(K):
        X = np.random.multivariate_normal(np.zeros(p), Sigma_true, size=n)
        data_machines.append(X)
    
    return data_machines, Sigma_true

def pca_comparison(l, n, K, p,T=1):
    """
    Compare the asymptotic efficiency of distributed PCA and pooled PCA.
    """
    # Calculate r automatically as the number of non-unit eigenvalues
    r = len(l)
    
    # Generate the dataset
    data_machines, Sigma_true = generate_data(l, n, K, p)
    
    # True population eigenspace (top r eigenvectors)
    eigvals_true = np.array([1 + li for li in l] + [1] * (p - len(l)))
    eigvecs_true = np.eye(p)  # Identity for simplicity, can be modified
    
    U_true = eigvecs_true[:, :r]  # Top r eigenvectors
    
    # Pooled PCA
    Up = pooled_pca(data_machines, r)
    
    # Distributed PCA
    Ud = distributed_pca(data_machines, r)

    #power PCA
    Ug = distributed_pca_power(data_machines, r,T=T)

    Ug_sigma = distributed_pca_power_sigma(data_machines, r,T=T,sigma=None)
    
    # Compute Frobenius errors
    error_up = frobenius_error(Up, U_true)
    error_ud = frobenius_error(Ud, U_true)
    error_ug = frobenius_error(Ug, U_true)
    error_ug_sigma = frobenius_error(Ug_sigma, U_true)

    return error_up, error_ud, error_ug, error_ug_sigma

