import random

import numpy as np
import pandas as pd

from sklearn import preprocessing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from matplotlib import pyplot as plt
import seaborn as sns
# %matplotlib inline

import os
for dirname, _, filenames in os.walk('./data_sets'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings("ignore")

# Data structure arrangement(数据结构整理)
folder = './data_sets/'
calendar = pd.read_csv(folder+'calendar.csv')
price = pd.read_csv(folder+'sell_prices.csv')
validation = pd.read_csv(folder+'test/sales_train_validation.csv')

d_cols = [col for col in validation.columns if col.startswith('d_')]
cc = validation[d_cols].stack().reset_index(level=1)
cc.columns = ['d','sales']
cc['day_int'] = cc.d.apply(lambda day:day.split('_')[1]).astype('int')
cc.sales = cc.sales.astype('int')
validation = validation.drop(d_cols, axis=1).join(cc)
#le = preprocessing.LabelEncoder()

price.store_id = price.store_id.astype('category')
price.item_id = price.item_id.astype('category')
price.wm_yr_wk = pd.to_numeric(price.wm_yr_wk, downcast='unsigned')
price.sell_price = pd.to_numeric(price.sell_price, downcast='float')

print(price.info())

calendar = calendar.fillna('NotEvent')

calendar.date = calendar.date.astype('datetime64[ns]')

calendar.wm_yr_wk = pd.to_numeric(calendar.wm_yr_wk, downcast='unsigned')
calendar.weekday = calendar.weekday.astype('category')
calendar.wday = pd.to_numeric(calendar.wday, downcast='unsigned')
calendar.month = pd.to_numeric(calendar.month, downcast='unsigned')
calendar.year = pd.to_numeric(calendar.year, downcast='unsigned')
calendar.d = calendar.d.astype('category')

calendar.event_name_1 = calendar.event_name_1.astype('category')
calendar.event_type_1 = calendar.event_type_1.astype('category')
calendar.event_name_2 = calendar.event_name_2.astype('category')
calendar.event_type_2 = calendar.event_type_2.astype('category')

calendar.snap_CA = calendar.snap_CA.astype('bool')
calendar.snap_TX = calendar.snap_TX.astype('bool')
calendar.snap_WI = calendar.snap_WI.astype('bool')

print(calendar.info())

validation.id = validation.id.astype('category')
validation.item_id = validation.item_id.astype('category')
validation.dept_id = validation.dept_id.astype('category')
validation.cat_id = validation.cat_id.astype('category')
validation.store_id = validation.store_id.astype('category')
validation.state_id = validation.state_id.astype('category')
validation.d = validation.d.astype('category')

validation.sales = pd.to_numeric(validation.sales, downcast='unsigned')
validation.day_int = pd.to_numeric(validation.day_int, downcast='unsigned')

print(validation.info())

validation = validation.merge(calendar, on='d', how='left').merge(price, on=['store_id','item_id','wm_yr_wk'], how='left')
del calendar, price
print(validation.info())

meta_df = pd.DataFrame({})

meta_df = meta_df._append([['item_id','id for item(商品id)','categorical(类别)','product(商品信息)','high(高)','']])
meta_df = meta_df._append([['dept_id','id for item dept(商品部id)','categorical(类别)','product(商品信息)','middle(中)','']])
meta_df = meta_df._append([['cat_id','id for item category(商品类别id)','categorical(类别)','product(商品信息)','middle(中)','']])
meta_df = meta_df._append([['sell_price','sell price for item(商品售价)','numerical(数值)','product(商品信息)','middle(中)','']])

meta_df = meta_df._append([['store_id','id for store(门店id)','categorical(类别)','store(门店信息)','middle(中)','']])
meta_df = meta_df._append([['state_id','id for state(门店所在州id)','categorical(类别)','store(门店信息)','middle(中)','']])

meta_df = meta_df._append([['weekday','day of week(星期几)','categorical(类别)','context(上下文信息)','low(低)','']])
meta_df = meta_df._append([['wday','day of week in number(星期几，数字形式)','categorical(类别)','context(上下文信息)','middle(中)','']])
meta_df = meta_df._append([['month','month(月份)','categorical(类别)','context(上下文信息)','middle(中)','']])
meta_df = meta_df._append([['year','year(年份)','categorical(类别)','context(上下文信息)','low(低)','']])

meta_df.columns = ['name','desc','type','segment','expectation','conclusion']
print(meta_df.sort_values(by='expectation'))

# From the target - sales(从目标特征开始分析)
print(validation.sales.describe())
# 最小值为0，意味着没有异常的负值
# 平均值为1.126，最大值为763，说明存在极值，这在销售中也较为常见

# sns.distplot(validation.query('sales < 10').sales)
print("偏移量: %f" % validation.sales.skew())
print("峰值: %f" % validation.sales.kurt())


# 分析Top5与目标的关系
plt.subplots(figsize=(25, 15))


var = 'sell_price'

item_rand = random.choice(validation.item_id.unique().tolist())
ax = plt.subplot(3,3,1)
ax.set_title(item_rand)
ax.set_xlabel('sell price')
ax.set_ylabel('sales')
tmp = validation.query('item_id == "'+item_rand+'"')
data = pd.concat([tmp.sales, tmp[var]], axis=1)
plt.scatter(x=data[var], y=data['sales'])

item_rand = random.choice(validation.item_id.unique().tolist())
ax = plt.subplot(3,3,2)
ax.set_title(item_rand)
ax.set_xlabel('sell price')
ax.set_ylabel('sales')
tmp = validation.query('item_id == "'+item_rand+'"')
data = pd.concat([tmp.sales, tmp[var]], axis=1)
plt.scatter(x=data[var], y=data['sales'])

item_rand = random.choice(validation.item_id.unique().tolist())
ax = plt.subplot(3,3,3)
ax.set_title(item_rand)
ax.set_xlabel('sell price')
ax.set_ylabel('sales')
tmp = validation.query('item_id == "'+item_rand+'"')
data = pd.concat([tmp.sales, tmp[var]], axis=1)
plt.scatter(x=data[var], y=data['sales'])

item_rand = random.choice(validation.item_id.unique().tolist())
ax = plt.subplot(3,3,4)
ax.set_title(item_rand)
ax.set_xlabel('sell price')
ax.set_ylabel('sales')
tmp = validation.query('item_id == "'+item_rand+'"')
data = pd.concat([tmp.sales, tmp[var]], axis=1)
plt.scatter(x=data[var], y=data['sales'])

item_rand = random.choice(validation.item_id.unique().tolist())
ax = plt.subplot(3,3,5)
ax.set_title(item_rand)
ax.set_xlabel('sell price')
ax.set_ylabel('sales')
tmp = validation.query('item_id == "'+item_rand+'"')
data = pd.concat([tmp.sales, tmp[var]], axis=1)
plt.scatter(x=data[var], y=data['sales'])

item_rand = random.choice(validation.item_id.unique().tolist())
ax = plt.subplot(3,3,6)
ax.set_title(item_rand)
ax.set_xlabel('sell price')
ax.set_ylabel('sales')
tmp = validation.query('item_id == "'+item_rand+'"')
data = pd.concat([tmp.sales, tmp[var]], axis=1)
plt.scatter(x=data[var], y=data['sales'])

item_rand = random.choice(validation.item_id.unique().tolist())
ax = plt.subplot(3,3,7)
ax.set_title(item_rand)
ax.set_xlabel('sell price')
ax.set_ylabel('sales')
tmp = validation.query('item_id == "'+item_rand+'"')
data = pd.concat([tmp.sales, tmp[var]], axis=1)
plt.scatter(x=data[var], y=data['sales'])

item_rand = random.choice(validation.item_id.unique().tolist())
ax = plt.subplot(3,3,8)
ax.set_title(item_rand)
ax.set_xlabel('sell price')
ax.set_ylabel('sales')
tmp = validation.query('item_id == "'+item_rand+'"')
data = pd.concat([tmp.sales, tmp[var]], axis=1)
plt.scatter(x=data[var], y=data['sales'])

item_rand = random.choice(validation.item_id.unique().tolist())
ax = plt.subplot(3,3,9)
ax.set_title(item_rand)
ax.set_xlabel('sell price')
ax.set_ylabel('sales')
tmp = validation.query('item_id == "'+item_rand+'"')
data = pd.concat([tmp.sales, tmp[var]], axis=1)
plt.scatter(x=data[var], y=data['sales'])

# 没有明显的趋势显示出价格越低销量越多，反倒是能看到部分价格高了销售也高了，所以这里可能存在市场的供需关系对价格的调节



store_rand = random.choice(validation.store_id.unique().tolist())
tmp = validation.query('store_id == "'+store_rand+'"')
f, ax = plt.subplots(figsize=(25, 6))
tmp.groupby('item_id').mean()['sales'].sample(50).plot(kind='bar')
ax.set_title('item vs sales')
ax.set_xlabel('item')
ax.set_ylabel('sales')

store_rand = random.choice(validation.store_id.unique().tolist())
tmp = validation.query('store_id == "'+store_rand+'"')
f, ax = plt.subplots(figsize=(25, 6))
tmp.groupby('month').mean()['sales'].plot(kind='bar')
ax.set_title('month vs sales')
ax.set_xlabel('month')
ax.set_ylabel('sales')

store_rand = random.choice(validation.store_id.unique().tolist())
tmp = validation.query('store_id == "'+store_rand+'"')
f, ax = plt.subplots(figsize=(25, 6))
tmp.groupby('weekday').mean()['sales'].plot(kind='bar')
ax.set_title('weekday vs sales')
ax.set_xlabel('weekday')
ax.set_ylabel('sales')

tmp = validation.sample(100000)
f, ax = plt.subplots(figsize=(12, 6))
tmp.groupby('state_id').mean()['sales'].plot(kind='bar')
ax.set_title('state vs sales')
ax.set_xlabel('state')
ax.set_ylabel('sales')

# 通过上面的分析可以看出
# 商品与销售的关系是明显的，这也符合业务常识
# 月份与销量的影响接近于0
# 星期几与销量的影响大于月份
# 州与销量之间也有一定的影响

#分类客观分析
for col in validation.columns:
    if str(validation[col].dtype) == 'category':
        validation[col] = validation[col].cat.codes

print(validation.info())


# 数据的时序特征
sample = validation.query('store_id==0 and item_id==0')[['date','sales']]
sample = sample.set_index('date')
plt.subplots(figsize=(25, 5))
plt.plot(sample.asfreq('w').index, sample.asfreq('w').values)
plt.title('store_id==0 & item_id==0')

# 延时数据对比
plt.subplots(figsize=(25, 5))
plt.plot(sample.asfreq('M').index, sample.asfreq('M').values)
plt.plot(sample.asfreq('M').shift(7).index, sample.shift(7).asfreq('M').values)
plt.legend(['sales','sales with lag 7'])
plt.title('shift and lag')
plt.show()

# 销售额数据对比
plt.subplots(figsize=(25, 5))
(sample.asfreq('M').sales - sample.asfreq('M').sales.shift()).plot()
(sample.asfreq('M').sales - sample.asfreq('M').sales.shift(3)).plot()
(sample.asfreq('M').sales - sample.asfreq('M').sales.shift(6)).plot()
plt.legend(['Change with 1 month','Change with 3 month','Change with 6 month'])
plt.title('change')
plt.show()


# 滑窗分时数据
plt.subplots(figsize=(25, 10))
plt.title('rolling window')

ax = plt.subplot(3,1,1)
rolling_sample = sample.sales.rolling('7D').mean()
sample.sales.plot()
rolling_sample.plot()
plt.legend(['sales','sales with rolling 7 day'])

ax = plt.subplot(3,1,2)
rolling_sample = sample.sales.rolling('30D').mean()
sample.sales.plot()
rolling_sample.plot()
plt.legend(['sales','sales with rolling 30 day'])

ax = plt.subplot(3,1,3)
rolling_sample = sample.sales.rolling('90D').mean()
sample.sales.plot()
rolling_sample.plot()
plt.legend(['sales','sales with rolling 90 day'])


# 自相关数据
plt.subplots(figsize=(15, 5))
ax = plt.subplot(1,1,1)
plot_acf(sample.sales,lags=25,title="Autocorrelation", ax=ax).show()

# 偏自相关数据
plt.subplots(figsize=(15, 5))
ax = plt.subplot(1,1,1)
plot_pacf(sample.sales,lags=25,title="Partial Autocorrelation", ax=ax).show()