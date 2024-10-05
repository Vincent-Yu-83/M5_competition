# <p align="center">M5 competition</p>


M5 competition项目是根据沃尔玛美国各个门店的销售数据（开放数据集），来对两个28天的时间段内的门店&商品的销售额进行预测


<br/>

## Table of Contents

#### 1. 数据集介绍
* [dropout_pytorch.ipynb](MLP/dropout_pytorch.ipynb) 
* calendar.csv - Contains information about the dates on which the products are sold.(提供商品销售的日期信息)
* sales_train_validation.csv - Contains the historical daily unit sales data per product and store [d_1 - d_1913](验证数据集，包含商品信息、门店信息、对应每天的销售额)
* sample_submission.csv - The correct format for submissions. Reference the Evaluation tab for more info.(提交文件样本文件)
* sell_prices.csv - Contains information about the price of the products sold per store and date.(门店的商品在某一天的价格信息)
* sales_train_evaluation.csv - Available once month before competition deadline. Will include sales [d_1 - d_1941](评价数据集，在竞赛结束前一个月提供，格式与验证数据集一致)

* 数据集下载地址：[https://www.wen1tech.com/datasets/M5_datasets.zip](https://www.wen1tech.com/datasets/M5_datasets.zip)


#### 2. [analysis.py](analysis.py) 
* 基础数据分析

#### 3. [feature_engineering.ipynb](feature_engineering.ipynb)
* 数据特征分析和提取

#### 4. [LightGBM_simple.py](LightGBM_simple.py) 
* LGBM简单实现

#### 5. [LGBM.py](LGBM.py) 
* LGBM分析数据并输出结果

#### 6. [LightGBM_poisson.ipynb](LightGBM_poisson.ipynb) 
* LGBM实现泊松分布计算

#### 7. [LightGBM2.ipynb](LightGBM2.ipynb) 
* LGBM分析商品类别分布和预测

<br/>

## Getting Started
```bash
$ git clone https://github.com/Vincent-Yu-83/M5_competition.git
$ pip install jupyterlab
$ pip install ipykernel
$ pip install pytorch
```

<br/>

## Dependencies
* [Python 3.11](https://www.continuum.io/downloads)
* [PyTorch 2.2.2](http://pytorch.org/)
