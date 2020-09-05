#! -*- coding:utf-8 -*-
import json
import numpy as np
from tqdm import tqdm
import time
import logging
from sklearn.model_selection import StratifiedKFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
import pandas as pd
import re
import jieba
from keras.utils import to_categorical
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score, roc_auc_score
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import sys
import ipykernel

pd.set_option('display.max_columns', None)


base = 'F:/Study_documents/PycharmProjects/chinese_L-12_H-768_A-12/'
config_path = base + 'bert_config.json'
checkpoint_path = base + 'bert_model.ckpt'
dict_path = base + 'vocab.txt'
MAX_LEN = 150

num_class = 3

token_dict = {}
with open(dict_path, 'r', encoding='utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)  # 给每个字按顺序加编号，len(token_dict)字典每加1个词便加1
tokenizer = Tokenizer(token_dict)

file_path = './log/'
# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sub = pd.read_csv('data/test.csv')

train_comment = train['微博中文内容'].values
test_comment = test['微博中文内容'].values

labels = train['情感倾向'].astype(int).values
labels += 1
'''
def f1_loss(y_true, y_pred):
    # y_true:真实标签0或者1；y_pred:为正类的概率
    loss = f1_score(y_true, y_pred, average='micro')
    return -loss
'''
def seq_padding(X, padding=0):  # 以X中最长句子为标准进行补齐
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data[0]) // self.batch_size
        if len(self.data[0]) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            X, y = self.data

            idxs = list(range(len(self.data[0])))
            np.random.shuffle(idxs)
            T, T_, Y = [], [], []
            for c, i in enumerate(idxs):
                d = X[i]
                text = d[:MAX_LEN]
                t, t_ = tokenizer.encode(first=text)
                T.append(t)
                T_.append(t_)
                Y.append(y[i])
                if len(T) == self.batch_size or i == idxs[-1]:
                    Y = np.array(Y)
                    T = seq_padding(T)
                    T_ = seq_padding(T_)
                    yield [T, T_], Y
                    T, T_, Y = [], [], []


def get_model():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    for l in bert_model.layers:
        l.trainable = True

    T1 = Input(shape=(None,))
    T2 = Input(shape=(None,))

    T = bert_model([T1, T2])

    T = Lambda(lambda x: x[:, 0])(T)  # 取第0列向量，即CLS

    output = Dense(num_class, activation='softmax')(T) # 多分类时要改成softmax

    model = Model([T1, T2], output)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        metrics=['accuracy']
    )
    model.summary()  # 输出参数Param计算过程
    return model


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)


def predict(data):
    prob = np.zeros((len(data), num_class), dtype=np.float32)
    val_x1 = data
    for i in tqdm(range(len(val_x1))):
        X = val_x1[i]
        text = X[:MAX_LEN]
        t1, t1_ = tokenizer.encode(first=text)
        T1, T1_ = np.array([t1]), np.array([t1_])
        _prob = model.predict([T1, T1_])  # [[0.27954674 0.2992469  0.42120633]]
        prob[i] = _prob[0]
    return prob

def save_epoch_result(res):
    temp = np.argmax(res, axis=1)
    for i in range(len(temp)):
       temp[i] -= 1
    sub['y'] = temp
    sub[['id', 'y']].to_csv('./result/bert_{}.csv'.format(fold), index=False)

oof_train = np.zeros((len(train), num_class), dtype=np.float32)
oof_test = np.zeros((len(test), num_class), dtype=np.float32)


for fold, (train_index, valid_index) in enumerate(skf.split(train_comment, labels)):  # skf.split必须要求一维数组，所以labels在循环里面one_hot
    logger.info('================     fold {}        ==============='.format(fold))
    x1 = train_comment[train_index]
    y = to_categorical(labels[train_index], 3)  # one_hot编码

    val_x1 = train_comment[valid_index]
    val_y = to_categorical(labels[valid_index], 3)

    train_D = data_generator([x1, y])
    valid_D = data_generator([val_x1, val_y])

    model = get_model()
    model_weight_filepath = './models/bert{}.w'.format(fold)
    earlystopping = EarlyStopping(monitor='val_acc', verbose=1, patience=2)  # 若2个epoch没有提高则early_stopping
    reducelronplateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.8,
                                          patience=1)  # val_acc在2个epoch没有提高，则学习率下降0.5 （new_lr） = lr * factor
    checkpoint = ModelCheckpoint(filepath=model_weight_filepath, monitor='val_acc',
                                 verbose=1, save_best_only=True, save_weights_only=True, mode='max',
                                 period=1)  # save_best_only：当设置为True时，监测值有改进时才会保存当前的模型；在这里若epochs有改进，则覆盖前一个。

    model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),  # 从generator产生的步骤的总数（样本批次总数）。通常情况下，应该等于数据集的样本数量除以批量的大小。
        epochs=5,
        validation_data=valid_D.__iter__(),
        validation_steps=len(valid_D),
        callbacks=[earlystopping, reducelronplateau, checkpoint])

    model.load_weights('./models/bert{}.w'.format(fold))
    oof_train[valid_index] = predict(val_x1)
    res = predict(test_comment)  #[[0.27954674 0.2992469  0.42120633]]
    oof_test += res
    save_epoch_result(res)

    del model
    K.clear_session()

np.save('./result/train.npy', oof_train)
np.save('./result/test_orgin.npy', oof_test)

x1 = np.array(oof_train)
from scipy import optimize
def fun(x):
    tmp = np.hstack([x[0] * x1[:, 0].reshape(-1, 1), x[1] * x1[:, 1].reshape(-1, 1), x[2] * x1[:, 2].reshape(-1, 1)])
    return -f1_score(labels, np.argmax(tmp, axis= 1), average='micro')
x0 = np.asarray((0,0,0))
res = optimize.fmin_powell(fun, x0)

xx_score = f1_score(labels, np.argmax(oof_train,axis=1), average='micro')
print("原始f1_score",xx_score)
xx_cv = f1_score(labels, np.argmax(oof_train*res,axis=1), average='micro')
print("修正后f1_score",xx_cv)

oof_test /= 5

oof_test_orgin = np.argmax(oof_test,axis=1)  # 返回list格式，如[0,1,2,2,0,1.....]
oof_test_optimizer = np.argmax(oof_test*res,axis=1)

for i in range(len(oof_test_orgin)):
    oof_test_orgin[i] -= 1


for i in range(len(oof_test_optimizer)):
    oof_test_optimizer[i] -= 1

np.save('./result/test_optimizer.npy', oof_test*res)


sub['y'] = oof_test_orgin
sub[['id', 'y']].to_csv('./result/bert_{}.csv'.format(xx_score), index=False)

sub['y'] = oof_test_optimizer
sub[['id', 'y']].to_csv('./result/bert_{}.csv'.format(xx_cv), index=False)

