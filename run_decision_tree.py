import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import load_data

feature_list = load_data.get_initial_feature_list()
df = pd.read_csv(load_data.path)
y = df[load_data.target]
df = df[feature_list]

str_columns = [column for column in df.columns if column not in load_data.dense_features_cols]

for feature in str_columns:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])

x = df[feature_list]

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)

clf = tree.DecisionTreeClassifier(class_weight='balanced', max_depth=None)  # 载入决策树分类模型
clf = clf.fit(xtrain.values, ytrain.values)  # 决策树拟合，得到模型
score = clf.score(xtest.values, ytest.values)  # 返回预测的准确度

clf_with_max_depth = tree.DecisionTreeClassifier(class_weight='balanced', max_depth=5)  # 载入决策树分类模型
clf_with_max_depth = clf_with_max_depth.fit(xtrain.values, ytrain.values)  # 决策树拟合，得到模型
score_new = clf_with_max_depth.score(xtest.values, ytest.values)  # 返回预测的准确度

print(" 决策树_未剪枝:{} \n".format(score), "决策树_剪枝:{}".format(score_new))


clf.feature_importances_
[*zip(feature_list, clf.feature_importances_)]

# 横向柱状图
plt.barh(np.arange(len(feature_list)), clf_with_max_depth.feature_importances_, align='center')
plt.yticks(np.arange(len(feature_list)), feature_list, fontsize=5)
plt.xticks(fontsize=10)
plt.xlabel('Importances', fontsize=15)
plt.xlim(0, 1)
plt.show()


# 未剪枝决策树 预测测试集
y_test_proba = clf.predict_proba(xtest)
false_positive_rate, recall, thresholds = roc_curve(ytest, y_test_proba[:, 1])
# 未剪枝决策树AUC
roc_auc = auc(false_positive_rate, recall)

# 剪枝决策树 预测测试集
y_test_proba_new = clf_with_max_depth.predict_proba(xtest)
false_positive_rate_new, recall_new, thresholds_new = roc_curve(ytest, y_test_proba_new[:, 1])
# 剪枝决策树AUC
roc_auc_new = auc(false_positive_rate_new, recall_new)

# 画出两个模型ROC曲线
plt.plot(false_positive_rate, recall, color='blue', label='AUC_orig=%0.3f' % roc_auc)
plt.plot(false_positive_rate_new, recall_new, color='orange', label='AUC_jianzhi=%0.3f' % roc_auc_new)
plt.legend(loc='best', fontsize=15, frameon=False)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()

# 定义空列表，用来存放每一个树的深度所对应的AUC值
auc_test = []
for i in range(20):
    clf = tree.DecisionTreeClassifier(class_weight='balanced', max_depth=i + 1)

    clf = clf.fit(xtrain, ytrain)
    y_test_proba = clf.predict_proba(xtest)
    false_positive_rate, recall, thresholds = roc_curve(ytest, y_test_proba[:, 1])
    roc_auc = auc(false_positive_rate, recall)
    auc_test.append(roc_auc)

plt.plot(range(1, 21), auc_test, color="red", label="max_depth")
plt.legend()
plt.show()

# total = len(ytest)
# correct = 0
# merged_num = 0
# abandoned_num = 0
# for row_id in range(xtest.shape[0]):
#     x_item = xtest.iloc[row_id]
#     y_item = ytest.iloc[row_id]
#     y_predict = clf.predict(x_item.values.reshape(1, -1))
#     if y_item == 1 and y_predict == 1:
#         merged_num += 1
#     if y_item == 0 and y_predict == 0:
#         abandoned_num += 1
#     correct += (y_predict == y_item).sum().item()  # 如果预测值和真实值相同， 则为true=1,  求和
#
# print(merged_num)
# print(abandoned_num)
# print(correct)
# print(total)
# print('Accuracy : %d %%' % (100 * correct / total))

