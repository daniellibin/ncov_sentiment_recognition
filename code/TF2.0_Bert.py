import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.utils import to_categorical
import os
from transformers import *
print(tf.__version__)


PATH = './'
BERT_PATH = './'
MAX_SEQUENCE_LENGTH = 140
input_categories = '微博中文内容'
output_categories = '情感倾向'

df_train = pd.read_csv(PATH+'nCoV_100k_train.labled.csv',engine ='python')
df_train = df_train[df_train[output_categories].isin(['-1','0','1'])]
df_test = pd.read_csv(PATH+'nCov_10k_test.csv',engine ='python')
df_sub = pd.read_csv(PATH+'submit_example.csv')
print('train shape =', df_train.shape)
print('test shape =', df_test.shape)


def _convert_to_transformer_inputs(instance, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""

    def return_id(str1, truncation_strategy, length):
        inputs = tokenizer.encode_plus(str1,
                                       add_special_tokens=True,
                                       max_length=length,
                                       truncation_strategy=truncation_strategy)

        input_ids = inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)

        return [input_ids, input_masks, input_segments]

    input_ids, input_masks, input_segments = return_id(
        instance, 'longest_first', max_sequence_length)

    return [input_ids, input_masks, input_segments]


def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    for instance in tqdm(df[columns]):
        ids, masks, segments = \
            _convert_to_transformer_inputs(str(instance), tokenizer, max_sequence_length)

        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)

    return [np.asarray(input_ids, dtype=np.int32),
            np.asarray(input_masks, dtype=np.int32),
            np.asarray(input_segments, dtype=np.int32)
            ]

tokenizer = BertTokenizer.from_pretrained(BERT_PATH+'bert-base-chinese-vocab.txt')
inputs = compute_input_arrays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
test_inputs = compute_input_arrays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)


def compute_output_arrays(df, columns):
    return np.asarray(df[columns].astype(int) + 1)
outputs = compute_output_arrays(df_train, output_categories)


def create_model():
    input_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    config = BertConfig.from_pretrained(BERT_PATH + 'bert-base-chinese-config.json')
    config.output_hidden_states = False
    bert_model = TFBertModel.from_pretrained(
        BERT_PATH + 'bert-base-chinese-tf_model.h5', config=config)
    # if config.output_hidden_states = True, obtain hidden states via bert_model(...)[-1]
    embedding = bert_model(input_id, attention_mask=input_mask, token_type_ids=input_atn)[0]

    x = tf.keras.layers.GlobalAveragePooling1D()(embedding)
    x = tf.keras.layers.Dropout(0.15)(x)
    x = tf.keras.layers.Dense(3, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=[input_id, input_mask, input_atn], outputs=x)

    return model


gkf = StratifiedKFold(n_splits=5).split(X=df_train[input_categories].fillna('-1'),
                                        y=df_train[output_categories].fillna('-1'))

valid_preds = []
test_preds = []
for fold, (train_idx, valid_idx) in enumerate(gkf):
    train_inputs = [inputs[i][train_idx] for i in range(len(inputs))]
    train_outputs = to_categorical(outputs[train_idx])

    valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
    valid_outputs = to_categorical(outputs[valid_idx])

    K.clear_session()
    model = create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', 'mae'])
    model.fit(train_inputs, train_outputs, validation_data=[valid_inputs, valid_outputs], epochs=1, batch_size=128)
    # model.save_weights(f'bert-{fold}.h5')
    valid_preds.append(model.predict(valid_inputs))
    test_preds.append(model.predict(test_inputs))


oof_train = np.vstack([pred for pred in valid_preds])
oof_test = np.average(test_preds, axis=0)

np.save('./result/train.npy', oof_train)
np.save('./result/test_orgin.npy', oof_test)


x1 = np.array(oof_train)
from scipy import optimize
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score, roc_auc_score
def fun(x):
    tmp = np.hstack([x[0] * x1[:, 0].reshape(-1, 1), x[1] * x1[:, 1].reshape(-1, 1), x[2] * x1[:, 2].reshape(-1, 1)])
    return -f1_score(outputs, np.argmax(tmp, axis= 1), average='macro')
x0 = np.asarray((0,0,0))
res = optimize.fmin_powell(fun, x0)

xx_score = f1_score(outputs, np.argmax(oof_train,axis=1), average='macro')
print("原始f1_score",xx_score)
xx_cv = f1_score(outputs, np.argmax(oof_train*res,axis=1), average='macro')
print("修正后f1_score",xx_cv)


oof_test_orgin = np.argmax(oof_test,axis=1)  # 返回list格式，如[0,1,2,2,0,1.....]
oof_test_optimizer = np.argmax(oof_test*res,axis=1)


oof_test_orgin -= 1
oof_test_optimizer -= 1

np.save('./result/test_optimizer.npy', oof_test*res)


df_sub['y'] = oof_test_orgin
df_sub[['id', 'y']].to_csv('./result/bert_{}.csv'.format(xx_score), index=False)

df_sub['y'] = oof_test_optimizer
df_sub[['id', 'y']].to_csv('./result/bert_{}.csv'.format(xx_cv), index=False)