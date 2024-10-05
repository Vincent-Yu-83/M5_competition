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

#### 2. [analysis.py](analysis.py) 
* [BN.ipynb](CNN/BN.ipynb) 
* [CNN-Explainer.html](CNN/CNN-Explainer.html) 
* [cnn.ipynb](CNN/cnn.ipynb) 
* [DenseNet.ipynb](CNN/DenseNet.ipynb) 
* [GoogLeNet.ipynb](CNN/GoogLeNet.ipynb) 
* [imgaug.html](CNN/imgaug.html) 
* [LeNet.ipynb](CNN/LeNet.ipynb) 
* [NiN.ipynb](CNN/NiN.ipynb) 
* [ResNet.ipynb](CNN/ResNet.ipynb) 
* [VGG.ipynb](CNN/VGG.ipynb)
#### 3. RNN
* [dataset.ipynb](RNN/dataset.ipynb) 
* [GRU.ipynb](RNN/GRU.ipynb) 
* [LSTM,.ipynb](RNN/LSTM,.ipynb) 
* [RNN_pytorch.ipynb](RNN/RNN_pytorch.ipynb) 
* [RNN.ipynb](RNN/RNN.ipynb) 
* [seq.ipynb](RNN/seq.ipynb) 
* [text.ipynb](RNN/text.ipynb) 
* [transfer.ipynb](RNN/transfer.ipynb)
#### 4. computer_vision
* [anchor_box.ipynb](computer_vision/anchor_box.ipynb) 
* [FCN.ipynb](computer_vision/FCN.ipynb) 
* [fine_tuning.ipynb](computer_vision/fine_tuning.ipynb) 
* [image_ augmentation.ipynb](<computer_vision/image_ augmentation.ipynb>) 
* [object_recognition.ipynb](computer_vision/object_recognition.ipynb) 
* [style_transfer.ipynb](computer_vision/style_transfer.ipynb)
#### 5. transformers
* [bertviz.ipynb](transformers/bertviz.ipynb) 
* [GloVe.ipynb](transformers/GloVe.ipynb) 
* [natural_language_inference_attention.ipynb](transformers/natural_language_inference_attention.ipynb) 
* [natural_language_inference_bert.ipynb](transformers/natural_language_inference_bert.ipynb) 
* [natural_language_inference_SNLI.ipynb](transformers/natural_language_inference_SNLI.ipynb) 
* [pretrain_bert_datasets.ipynb](transformers/pretrain_bert_datasets.ipynb) 
* [pretrain_bert.ipynb](transformers/pretrain_bert.ipynb) 
* [sentiment_analysis_BiRNN.ipynb](transformers/sentiment_analysis_BiRNN.ipynb) 
* [sentiment_analysis_CNN.ipynb](transformers/sentiment_analysis_CNN.ipynb) 
* [sentiment_analysis_datasets.ipynb](transformers/sentiment_analysis_datasets.ipynb) 
* [transformer.ipynb](transformers/transformer.ipynb) 
* [word2vec.ipynb](transformers/word2vec.ipynb)
#### 6. tools
* [tool_pytorch_017.py](tools/tool_pytorch_017.py)

<br/>

## Getting Started
```bash
$ git clone https://github.com/Vincent-Yu-83/neural_network.git
$ pip install jupyterlab
$ pip install ipykernel
$ pip install pytorch
```

<br/>

## Dependencies
* [Python 3.8](https://www.continuum.io/downloads)
* [PyTorch 2.2.2](http://pytorch.org/)
