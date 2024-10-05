import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns


import os
for dirname, _, filenames in os.walk('./data_sets'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import gc
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# Data structure arrangement(数据结构整理)
folder = './data_sets/'
df_cal = pd.read_csv(folder+'calendar.csv')
df_price = pd.read_csv(folder+'sell_prices.csv')
df_eval = pd.read_csv(folder+'train/sales_train_evaluation.csv')
df_sample_output = pd.read_csv(folder+'sample_submission.csv')

holiday = ['NewYear', 'OrthodoxChristmas', 'MartinLutherKingDay', 'SuperBowl', 'PresidentsDay', 'StPatricksDay', 'Easter', 'Cinco De Mayo', 'IndependenceDay', 'EidAlAdha', 'Thanksgiving', 'Christmas']
weekend = ['Saturday', 'Sunday']

df_cal['is_holiday_1'] = df_cal['event_name_1'].apply(lambda x : 1 if x in holiday else 0 )
df_cal['is_holiday_2'] = df_cal['event_name_1'].apply(lambda x : 1 if x in holiday else 0 )
df_cal['is_holiday'] = df_cal[['is_holiday_1','is_holiday_2']].max(axis=1)
df_cal['is_weekend'] = df_cal['weekday'].apply(lambda x : 1 if x in weekend else 0 )

df_cal = df_cal.drop(['weekday', 'wday', 'month', 'year', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'], axis='columns')
del_col = []

for x in range(1851):
    del_col.append('d_' + str(x+1))
    
df_eval = df_eval.drop(del_col, axis='columns')
df_eval = df_eval.melt(['id','item_id','dept_id','cat_id','store_id','state_id'], var_name='d', value_name='qty')
print(df_eval.shape)
df_eval.head()

df_eval = pd.merge(df_eval, df_cal, how='left', on='d')
df_eval = pd.merge(df_eval, df_price, how='left', on=['item_id', 'wm_yr_wk', 'store_id'])

df_eval_test = df_eval.query('d == "d_1852"')
df_eval_test = df_eval_test[['id', 'store_id', 'item_id', 'dept_id', 'cat_id', 'state_id', 'd', 'qty', 'sell_price']]
df_eval_test['qty'] = df_eval_test['d'].apply(lambda x: int(x.replace(x, '0')))
tmp_df = df_eval_test


for x in range(28):
    df_eval_test = df_eval_test._append(tmp_df)
    
df_eval_test = df_eval_test.reset_index(drop=True)
lst_d = []
i = 0
lst_index = df_eval_test.index

for x in lst_index:
    lst_d.append('d_' + str(((lst_index[i]) // 30490) + 1942))
    i = i + 1
    
df_eval_test['d'] = lst_d
df_eval_test = pd.merge(df_eval_test, df_cal, how='left', on='d')
df_eval_test = pd.merge(df_eval_test, df_price, how='left', on=['item_id', 'wm_yr_wk', 'store_id'])


# 数据整理
del tmp_df
gc.collect()
df_eval = pd.get_dummies(data=df_eval, columns=['dept_id', 'cat_id', 'store_id', 'state_id'])
df_eval_test = pd.get_dummies(data=df_eval_test, columns=['dept_id', 'cat_id', 'store_id', 'state_id'])
df_eval_test = df_eval_test.drop(['sell_price_x', 'snap_CA', 'snap_TX', 'snap_WI'], axis='columns')
df_eval_test = df_eval_test.rename(columns={'sell_price_y': 'sell_price'})
df_eval = df_eval.drop(['snap_CA', 'snap_TX', 'snap_WI'], axis='columns')                                   


target_col = 'qty'
exclude_cols = ['id', 'item_id', 'd', 'date', 'wm_yr_wk']
feature_cols = [col for col in df_eval.columns if col not in exclude_cols]
y = np.array(df_eval[target_col])
X = np.array(df_eval[feature_cols])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 导入LGBM模型
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test)

params = {
        'n_jobs': -1,
        'boosting_type': 'gbdt',
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 64,
        "learning_rate" : 0.005,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.5,
        "bagging_frequency" : 6,
        "bagging_seed" : 42,
        "verbosity" : 1,
        "seed": 42,
        }
    # {'boosting_type': 'goss',
    #       'max_depth' : -1,
    #       'objective': 'regression',
    #       'nthread': 3, # Updated from nthread
    #       'num_leaves': 64,
    #       'learning_rate': 0.05,
    #       'max_bin': 512,
    #       'subsample_for_bin': 200,
    #       'subsample': 1,
    #       'subsample_freq': 1,
    #       'colsample_bytree': 0.8,
    #       'reg_alpha': 5,
    #       'reg_lambda': 10,
    #       'min_split_gain': 0.5,
    #       'min_child_weight': 1,
    #       'min_child_samples': 5,
    #       'scale_pos_weight': 1,
    #       'num_class' : 1,
    #       'metric' : 'rmse'}
    
evals_result = {}

model = lgb.train(params, lgb_train, 5000, 
                      valid_sets=[lgb_eval], 
                    #   early_stopping_rounds=1500,
                    #   verbose_eval=50,
                    #   evals_result=evals_result
                      )

pred = model.predict(df_eval_test[feature_cols])
df_eval_test['pred_qty'] = pred
predictions = df_eval_test[['id', 'date', 'pred_qty']]
predictions = pd.pivot(predictions, index = 'id', columns = 'date', values = 'pred_qty').reset_index()
predictions
predictions = predictions.drop(predictions.columns[1], axis=1)
predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]
x = 2744099 + 1 - 853720
df_val = df_eval[x:]

# 数据输出
# predictions_v = df_val[['id', 'date', 'qty']]
# predictions_v = pd.pivot(predictions_v, index = 'id', columns = 'date', values = 'qty').reset_index()
# predictions_v['id'] = predictions['id'].apply(lambda x: x.replace('evaluation', 'validation'))
# predictions_v.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]
# predictions_concat = pd.concat([predictions, predictions_v], axis=0)

# predictions_concat.to_csv('submission.csv', index=False)