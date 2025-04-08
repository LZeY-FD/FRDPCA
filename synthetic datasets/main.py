from algorithm import *

l = [3.75,3.5,3.25] #strong
#l = [3.25,3,2.75] #moderate
#l = [2.75,2.5,2.25] #weak

r = len(l)

pooled_PCA_num = []
dist_PCA_num = []
power_PCA_num = []


pooled_PCA_rmt = []
dist_PCA_rmt = []

p = 200
K = 30
n_list = [100,125,150,175,200,225,250,275,300,325,350,375,400]

for n in n_list:
    N = n*K
    

    error_up_list = []
    error_ud_list = []
    error_ug_list = []

    for _ in range(100):
        error_up, error_ud,error_ug = pca_comparison(l, n, K, p)
        error_up_list.append(error_up)
        error_ud_list.append(error_ud)
        error_ug_list.append(error_ug)


    lambda_p_list = np.array([lambda_spike_RMT(K*n, p,i) for i in l])
    lambda_d_list = np.array([lambda_spike_RMT(n, p,i) for i in l])
    
    pooled_PCA_num.append(np.mean(error_up_list))
    dist_PCA_num.append(np.mean(error_ud_list))
    power_PCA_num.append(np.mean(error_ug_list))


    pooled_PCA_rmt.append(r-np.sum(lambda_p_list))
    dist_PCA_rmt.append(np.sum((1-lambda_d_list)/lambda_d_list)/K)

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(3,4), dpi=350)


sns.lineplot(x=n_list, y=pooled_PCA_rmt, label='Pooling RMT', color='tab:blue', linewidth=2.5, linestyle='--',zorder=1)
sns.lineplot(x=n_list, y=dist_PCA_rmt, label='1RD-RMT', color='tab:orange', linewidth=2.5, linestyle='--',zorder=1)

sns.scatterplot(x=n_list, y=pooled_PCA_num, label='Pooling PCA', color='tab:blue', s=30, marker='o', edgecolor='black', linewidth=0.5,zorder=2)
sns.scatterplot(x=n_list, y=dist_PCA_num, label='1RD-PCA', color='tab:orange', s=30, marker='s', edgecolor='black', linewidth=0.5,zorder=2)
sns.scatterplot(x=n_list, y=power_PCA_num, label='2RD-PCA', color='tab:green', s=30, marker='X', edgecolor='black', linewidth=0.5,zorder=2)


ax.set_xlabel('local sample size n', fontsize=12)
#ax.set_ylabel('PCA/Rate Value', fontsize=12)
ax.set_title(f'Strong spike (p={p})', fontsize=14)


plt.legend(fontsize=10, title_fontsize=12)

plt.ylim([0, 0.18])

plt.tight_layout()
plt.savefig(f'strong (p={p}).png')
