# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: ensemble_mean.py 
@Time: 2018/11/9 16:59
@Software: PyCharm 
@Description:
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import gc  # 垃圾回收
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from main import create_feature

# 加载数据
df_train = pd.read_csv('input/train.csv')
df_test = pd.read_csv('input/test.csv')
train_len = len(df_train)
df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
train, test = create_feature(df)
cols = [col for col in train.columns if col not in ['id', 'y']]
X_train, y_train, X_test = train[cols], train['y'], test[cols]

# 训练模型1 随机森林
print("rfc..")
rfc_1 = RandomForestClassifier(random_state=0, n_estimators=2000)
rfc_1.fit(X_train, y_train)
pred1 = rfc_1.predict_proba(X_test)[:, 1]

# 训练模型2
print("xgb..")
xgb = XGBClassifier(n_estimators=4000,
                    learning_rate=0.03,
                    num_leaves=30,
                    colsample_bytree=.8,
                    subsample=.9,
                    max_depth=7,
                    reg_alpha=.1,
                    reg_lambda=.1,
                    min_split_gain=.01,
                    min_child_weight=2,
                    verbose=True)
xgb.fit(X_train, y_train)
pred2 = xgb.predict_proba(X_test)[:, 1]

# 训练模型3 gbm
print("gbm..")
gbm = LGBMClassifier(n_estimators=4000,
                     learning_rate=0.03,
                     num_leaves=30,
                     colsample_bytree=.8,
                     subsample=.9,
                     max_depth=7,
                     reg_alpha=.1,
                     reg_lambda=.1,
                     min_split_gain=.01,
                     min_child_weight=2,
                     silent=-1,
                     verbose=-1, )
gbm.fit(X_train, y_train)
pred3 = gbm.predict_proba(X_test)[:, 1]

y_test = np.average(np.array([pred1, pred2, pred3]), axis=0, weights=[0.1, 0.7, 0.2])
test['y'] = y_test
test[['id', 'y']].to_csv('result/ensemble_mean.csv', columns=None, header=False, index=False)
