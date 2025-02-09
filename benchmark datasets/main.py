import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from algorithm import *

def standardized(X):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

def split_data(X_train, K):
    return np.array_split(X_train, K)

def compute_ar(U, A):
    return np.linalg.norm(U @ (U.T @ A))**2 / np.linalg.norm(A)**2


def run_experiment_benchmark(data_list, T=10, k_ratio=1, r_ratio=0.2, r_max=20):
    results = []

    for data_path in data_list:
        df = pd.read_csv(f"datasets/{data_path}")
        
        X = df.iloc[:, :-1]  # 假设最后一列是标签列
        N, p = X.shape

        K = max(round(N / p * k_ratio), 1)
        r = min(int(r_ratio * p), r_max)  # r 不能超过 p

        X = standardized(X)

        result = run_experiment(X, T, K, r)
        results.append(result)

        print(f"Dataset {data_path} finished: K={K}, r={r}")

    df_result = pd.DataFrame(results, index=data_list)
    print(df_result)
    
    return df_result

data_list = ['clf_num/pol.csv','clf_num/default-of-credit-card-clients.csv','clf_num/eye_movements.csv',
        'clf_num/heloc.csv','clf_cat/eye_movements.csv',
        'clf_cat/albert.csv','clf_cat/default-of-credit-card-clients.csv',
        'clf_cat/road-safety.csv', 'reg_num/cpu_act.csv','reg_num/pol.csv',
         'reg_num/Ailerons.csv','reg_num/yprop_4_1.csv',
         'clf_num/MiniBooNE.csv','clf_num/jannis.csv',
             'clf_cat/covertype.csv','reg_num/superconduct.csv',
        'clf_num/Bioresponse.csv', 'reg_cat/topo_2_1.csv', 'reg_cat/Allstate_Claims_Severity.csv',
          'reg_cat/Mercedes_Benz_Greener_Manufacturing.csv'
         ]

kappa = 2
rho = 0.2
df_result = run_experiment_benchmark(data_list, T=10, k_ratio=kappa, r_ratio=rho, r_max=20)
main_plot(df_result, kappa=kappa,rho=rho)
