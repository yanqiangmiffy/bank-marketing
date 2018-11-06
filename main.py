# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: main.py 
@Time: 2018/11/5 17:11
@Software: PyCharm 
@Description:
"""
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
import lightgbm as lgb
pd.set_option('display.max_columns',100)
df_train=pd.read_csv('input/train.csv')
df_test=pd.read_csv('input/test.csv')


def add_poly_features(data,column_names):
    # 组合特征
    features=data[column_names]
    rest_features=data.drop(column_names,axis=1)
    poly_transformer=PolynomialFeatures(degree=2,interaction_only=False,include_bias=False)
    poly_features=pd.DataFrame(poly_transformer.fit_transform(features),columns=poly_transformer.get_feature_names(column_names))

    for col in poly_features.columns:
        rest_features.insert(1,col,poly_features[col])
    return rest_features


def process_label(train,test,cate_cols):
    # 数据预处理 类别编码
    cate_cols=['job','marital','education','default','housing','loan','contact','day','month','poutcome']
    for col in cate_cols:
        lb_encoder=LabelEncoder()
        train[col]=lb_encoder.fit_transform(train[col])
        test[col]=lb_encoder.transform(test[col]) # 这个步骤风险有点大，因为test的类别标签不一定都出现在train里面，这里比较幸运

    train = add_poly_features(train, cate_cols)
    test = add_poly_features(test, cate_cols)

    train = pd.get_dummies(train, columns=cate_cols)
    test = pd.get_dummies(test, columns=cate_cols)
    return train,test


def process_nums(train,test,num_cols):
    # 数据预处理 数值型数据
    num_cols=['age','balance','duration','campaign','pdays','previous']
    scaler=MinMaxScaler()
    train[num_cols] = scaler.fit_transform(train[num_cols].values)
    test[num_cols] = scaler.transform(test[num_cols].values)

    return train,test


def create_feature(train,test):
    # 数据预处理 类别编码
    cate_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day', 'month', 'poutcome']
    train, test=process_label(train,test,cate_cols)
    # 数据预处理 数值型数据
    num_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
    train, test=process_nums(train,test,num_cols)

    print(train.shape)
    print(test.shape)
    return train,test


# cv5 交叉验证
def evaluate_cv5_lgb(train_df, test_df, cols, test=False):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_test = 0
    oof_train = np.zeros((train_df.shape[0],))
    for i, (train_index, val_index) in enumerate(kf.split(train_df[cols])):
        X_train, y_train = train_df.loc[train_index, cols], train_df.y.values[train_index]
        X_val, y_val = train_df.loc[val_index, cols], train_df.y.values[val_index]

        lgb_train = lgb.Dataset(
            X_train, y_train)
        lgb_eval = lgb.Dataset(
            X_val, y_val,
            reference=lgb_train)
        print('开始训练......')

        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'auc', 'binary_logloss'},
            'learning_rate': 0.025,
            'num_leaves': 38,
            'min_data_in_leaf': 170,
            'bagging_fraction': 0.85,
            'bagging_freq': 1,
            'seed': 42
        }

        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=40000,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=50,
                        verbose_eval=False,
                        )
        y_pred = gbm.predict(X_val)
        if test:
            y_test = gbm.predict(test_df.loc[:, cols])
        oof_train[val_index] = y_pred
    auc = roc_auc_score(train_df.y.values, oof_train)
    y_test /= 5
    print('5 Fold auc:', auc)
    return  y_test


train,test=create_feature(df_train,df_test)
print(list(train.columns))
cols = [col for col in train.columns if col not in ['id','y']]
y_test=evaluate_cv5_lgb(train,test,cols,True)

test['y']=y_test
test[['id','y']].to_csv('result/01_lgb_cv5.csv',columns=None, header=False, index=False)