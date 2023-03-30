import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config.TrainConfig import *

# 绘制数据集label的分布图

# set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above)
sns.set(style="darkgrid")

# 加载示例数据集
# df = pd.read_csv(data_path)
# label分布图
# 设置shade=True参数添加阴影
# sns.kdeplot(df['time'], fill=True, clip=[0, 60])
# sns.kdeplot(df['avg_score'], fill=True, clip=[0, 120])
# plt.show()

# 特征相关性热力图
df = pd.read_csv(data_path)
df = df[target_labels]
df.rename(columns={'avg_score': 'score'}, inplace=True)
correlation = df.corr(method='spearman')
# plt.subplots(figsize=(24, 18))
plt.subplots(figsize=(4, 3))
sns.heatmap(correlation, vmax=1, vmin=-0.5, annot=True)
# sns.heatmap(correlation)
plt.savefig("../heatmap1")
plt.show()
