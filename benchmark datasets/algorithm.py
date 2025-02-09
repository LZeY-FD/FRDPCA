import numpy as np
import matplotlib.pyplot as plt


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


def distributed_pca(data_machines, r):
    """
    Perform distributed PCA by calculating the leading eigenvectors for each machine and combining them.
    """
    K = len(data_machines)
    
    # Compute sample covariance matrices for each machine
    cov_matrices = [np.cov(X.T) for X in data_machines]
    
    # Compute leading r eigenvectors for each machine
    Uk_list = []
    for Sigma_k in cov_matrices:
        eigvals, eigvecs = np.linalg.eigh(Sigma_k)
        sorted_indices = np.argsort(eigvals)[::-1]
        Uk = eigvecs[:, sorted_indices[:r]]
        Uk_list.append(Uk)
    
    # Sum of Uk * Uk^T and then average by K
    U_combined = np.zeros((Sigma_k.shape[0], Sigma_k.shape[0]))  # Initialize for sum
    for Uk in Uk_list:
        U_combined += Uk @ Uk.T
    
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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def Fnorm(A):
    return np.linalg.norm(A)

def compute_ar(U, A):
    return (Fnorm(U @ (U.T @ A)) ** 2) / (Fnorm(A) ** 2)

def run_experiment(X, N, K, r):
    train_errors_up = []
    test_errors_up = []
    train_errors_ud = []
    test_errors_ud = []
    train_errors_ug = []
    test_errors_ug = []

    for _ in range(N):
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=None)

        data_machines = split_data(X_train, K)
        
        Up = pooled_pca(data_machines, r)
        Ud = distributed_pca(data_machines, r)
        Ug = distributed_pca_power(data_machines, r, T=1)

        train_errors_up.append(compute_ar(Up, X_train.T))
        test_errors_up.append(compute_ar(Up, X_test.T))
        
        train_errors_ud.append(compute_ar(Ud, X_train.T))
        test_errors_ud.append(compute_ar(Ud, X_test.T))
        
        train_errors_ug.append(compute_ar(Ug, X_train.T))
        test_errors_ug.append(compute_ar(Ug, X_test.T))

    print("Results after {} experiments:".format(N))
    
    print(f"Pooled PCA (Up) - Train AR: {np.mean(train_errors_up):.4f}, Test AR: {np.mean(test_errors_up):.4f}")
    print(f"Distributed PCA (Ud) - Train AR: {np.mean(train_errors_ud):.4f}, Test AR: {np.mean(test_errors_ud):.4f}")
    print(f"Distributed PCA Power (Ug) - Train AR: {np.mean(train_errors_ug):.4f}, Test AR: {np.mean(test_errors_ug):.4f}")
    
    return {
        "Up_train": np.mean(train_errors_up),
        "Ud_train": np.mean(train_errors_ud),
        "Ug_train": np.mean(train_errors_ug),
        "Up_test": np.mean(test_errors_up),
        "Ud_test": np.mean(test_errors_ud),
        "Ug_test": np.mean(test_errors_ug)
    }

def split_data(X, K):
    n = X.shape[0]
    data_machines = []
    indices = np.random.permutation(n)
    split_size = n // K
    for i in range(K):
        start_idx = i * split_size
        if i == K - 1:
            end_idx = n
        else:
            end_idx = (i + 1) * split_size
        data_machines.append(X.iloc[indices[start_idx:end_idx]])
    return data_machines


def main_plot(df_result, kappa=0.5, rho=0.2):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=350)

    color_Ud = "#E69F00"  
    color_Ug = "#56B4E9"  
    marker_Ud = "o"       
    marker_Ug = "s"       

    def get_stats(up, ud, ug):
        avg_ratio_ud = np.mean(ud / up)
        avg_ratio_ug = np.mean(ug / up)
        pct_ud_lt_ug = np.mean(ud < ug) * 100
        return avg_ratio_ud, avg_ratio_ug, pct_ud_lt_ug

    avg_ratio_ud_train, avg_ratio_ug_train, pct_ud_lt_ug_train = get_stats(df_result["Up_train"], df_result["Ud_train"], df_result["Ug_train"])
    avg_ratio_ud_test, avg_ratio_ug_test, pct_ud_lt_ug_test = get_stats(df_result["Up_test"], df_result["Ud_test"], df_result["Ug_test"])

    ax1 = axes[0]
    ax1.scatter(df_result["Up_train"], df_result["Ud_train"], 
                 color=color_Ud, marker=marker_Ud, label=r"AR($\hat{U}^{(1)}$)", alpha=0.7, s=30)
    ax1.scatter(df_result["Up_train"], df_result["Ug_train"], 
                color=color_Ug, marker=marker_Ug, label=r"AR($\hat{U}^{(2)}$)", alpha=0.7, s=30)

    lims = [min(df_result["Up_train"].min(), df_result["Ud_train"].min(), df_result["Ug_train"].min()) * 0.95, 
            max(df_result["Up_train"].max(), df_result["Ud_train"].max(), df_result["Ug_train"].max()) * 1.05]
    ax1.plot(lims, lims, '--', color="gray", linewidth=2)  

    ax1.set_xlabel(
        r"Mean AR($\hat{U}^{(1)}$) / AR($\hat{U}^{P}$)" +f" = {avg_ratio_ud_train:.2%}\n"
        r"Mean AR($\hat{U}^{(2)}$) / AR($\hat{U}^{P}$)"+ f" = {avg_ratio_ug_train:.2%}"
    )
    ax1.set_ylabel(r"| AR($\hat{U}^{(1)}$) < AR($\hat{U}^{(2)}$)|"+f" = {pct_ud_lt_ug_train:.1f}%")

    ax1.set_title(rf"Train Set ($\kappa$={kappa}, $\rho$={rho})")
    ax1.legend()

    ax2 = axes[1]
    ax2.scatter(df_result["Up_test"], df_result["Ud_test"], 
                color=color_Ud, marker=marker_Ud, label=r"AR($\hat{U}^{(1)}$)", alpha=0.7, s=30)
    ax2.scatter(df_result["Up_test"], df_result["Ug_test"], 
                color=color_Ug, marker=marker_Ug, label=r"AR($\hat{U}^{(2)}$)", alpha=0.7, s=30)

    lims = [min(df_result["Up_test"].min(), df_result["Ud_test"].min(), df_result["Ug_test"].min()) * 0.95, 
            max(df_result["Up_test"].max(), df_result["Ud_test"].max(), df_result["Ug_test"].max()) * 1.05]
    ax2.plot(lims, lims, '--', color="gray", linewidth=2)

    ax2.set_xlabel(
        r"Mean AR($\hat{U}^{(1)}$) / AR($\hat{U}^{P}$)" +f" = {avg_ratio_ud_test:.2%}\n"
        r"Mean AR($\hat{U}^{(2)}$) / AR($\hat{U}^{P}$)"+ f" = {avg_ratio_ug_test:.2%}"
    )
    ax2.set_ylabel(r"| AR($\hat{U}^{(1)}$) < AR($\hat{U}^{(2)}$)|"+f" = {pct_ud_lt_ug_test:.1f}%")

    ax2.set_title(rf"Test Set ($\kappa$={kappa}, $\rho$={rho})")
    ax2.legend()

    plt.savefig(rf'kappa={kappa}_rho={rho}.png', bbox_inches="tight",dpi=350)

    plt.tight_layout()
    plt.show()