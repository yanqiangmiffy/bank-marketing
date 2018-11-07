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
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

pd.set_option('display.max_columns',100)
df_train=pd.read_csv('input/train.csv')
df_test=pd.read_csv('input/test.csv')
train_len = len(df_train)
df = pd.concat([df_train, df_test], axis=0, ignore_index=True, sort=False)


def add_poly_features(data,column_names):
    # 组合特征
    features=data[column_names]
    rest_features=data.drop(column_names,axis=1)
    poly_transformer=PolynomialFeatures(degree=2,interaction_only=False,include_bias=False)
    poly_features=pd.DataFrame(poly_transformer.fit_transform(features),columns=poly_transformer.get_feature_names(column_names))

    for col in poly_features.columns:
        rest_features.insert(1,col,poly_features[col])
    return rest_features


def process_label(df,cate_cols):
    # 数据预处理 类别编码
    cate_cols=['job','marital','education','default','housing','loan','contact','poutcome']
    # for col in cate_cols:
    #     lb_encoder=LabelEncoder()
    #     df[col]=lb_encoder.fit_transform(df[col])
    df = pd.get_dummies(df, columns=cate_cols)
    return df


def process_nums(df,num_cols):
    # 数据预处理 数值型数据
    # num_cols=['age','balance','duration','campaign','pdays','previous']
    # scaler=MinMaxScaler()
    # df[num_cols] = scaler.fit_transform(df[num_cols].values)

    # 定义函数
    def standardize_nan(x):
        # 标准化
        x_mean = np.nanmean(x)
        x_std = np.nanstd(x)
        return (x - x_mean) / x_std

    df["log_balance"] = np.log(df.balance - df.balance.min() + 1)
    df["log_duration"] = np.log(df.duration + 1)
    df["log_campaign"] = np.log(df.campaign + 1)
    df["log_pdays"] = np.log(df.pdays - df.pdays.min() + 1)
    df = df.drop(["balance", "duration", "campaign", "pdays"], axis=1)

    # month 文字列与数値的変換
    month_dict = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                  "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
    df["month_int"] = df["month"].map(month_dict)
    # month 与day 对 datetime 的変換
    data_datetime = df \
        .assign(ymd_str=lambda x: "2014" + "-" + x["month_int"].astype(str) + "-" + x["day"].astype(str)) \
        .assign(datetime=lambda x: pd.to_datetime(x["ymd_str"])) \
        ["datetime"].values
    # datetime  int 变换
    index = pd.DatetimeIndex(data_datetime)
    df["datetime_int"] = np.log(index.astype(np.int64))

    # 删除不要的列
    df = df.drop(["month", "day", "month_int"], axis=1)
    del data_datetime
    del index
    return df


def create_feature(df):
    # 数据预处理 类别编码
    cate_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day', 'month', 'poutcome']
    new_df=process_label(df,cate_cols)

    # 数据预处理 数值型数据
    num_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
    new_df=process_nums(new_df,num_cols)

    new_train,new_test=new_df[:train_len],new_df[train_len:]
    print(list(new_train.columns))
    print(new_train.shape)
    return new_train,new_test


# 调整参数
def tune_params(model,params,X,y):
    gsearch = GridSearchCV(estimator=model,param_grid=params, scoring='roc_auc')
    gsearch.fit(X, y)
    print(gsearch.cv_results_, gsearch.best_params_, gsearch.best_score_)
    return gsearch


# 特征重要性
def plot_fea_importance(classifier,X_train):
    plt.figure(figsize=(10,12))
    name = "xgb"
    indices = np.argsort(classifier.feature_importances_)[::-1][:40]
    g = sns.barplot(y=X_train.columns[indices][:40],
                    x=classifier.feature_importances_[indices][:40],orient='h')
    g.set_xlabel("Relative importance", fontsize=12)
    g.set_ylabel("Features", fontsize=12)
    g.tick_params(labelsize=9)
    g.set_title(name + " feature importance")
    plt.show()


def evaluate_cv5_lgb(train_df, test_df, cols, test=False):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # xgb = XGBClassifier()
    # params = {"learning_rate": [0.08, 0.1, 0.12],
    #           "max_depth": [6, 7],
    #           "subsample": [0.95, 0.98],
    #           "colsample_bytree": [0.6, 0.7],
    #           "min_child_weight": [3, 3.5, 3.8]
    #           }
    # xgb = tune_params(xgb,params,train_df[cols],train_df.y.values)

    y_test = 0
    oof_train = np.zeros((train_df.shape[0],))
    for i, (train_index, val_index) in enumerate(kf.split(train_df[cols])):
        X_train, y_train = train_df.loc[train_index, cols], train_df.y.values[train_index]
        X_val, y_val = train_df.loc[val_index, cols], train_df.y.values[val_index]
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
        xgb.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                early_stopping_rounds=100, eval_metric=['auc'], verbose=True)
        y_pred = xgb.predict_proba(X_val)[:,1]
        if test:
            y_test += xgb.predict_proba(test_df.loc[:, cols])[:,1]
        oof_train[val_index] = y_pred
        if i==0:
            plot_fea_importance(xgb,X_train)
    auc = roc_auc_score(train_df.y.values, oof_train)
    y_test /= 5
    print('5 Fold auc:', auc)
    return y_test


train,test=create_feature(df)
# print(list(train.columns))
cols = [col for col in train.columns if col not in ['id','y']]
# y_test=evaluate_cv5_lgb1(train,test,cols,True)
y_test=evaluate_cv5_lgb(train,test,cols,True)

test['y']=y_test
test[['id','y']].to_csv('result/01_lgb_cv5.csv',columns=None, header=False, index=False)