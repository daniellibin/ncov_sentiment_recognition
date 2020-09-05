from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import argparse

from sklearn.preprocessing import LabelEncoder
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader

from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel, BertConfig
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score

# parser = argparse.ArgumentParser(allow_abbrev=False)

# parser.add_argument("--seed", type=int, default=42, help="seed")
# parser.add_argument("--cnt", type=int, default=0)
# parser.add_argument("--cut", type=int, default=-1)

# arg = parser.parse_args()

device = torch.device('cuda')


train = pd.read_csv('data/nCoV_100k_train.labled.csv')
train = train[train["情感倾向"].isin(['-1', '0', '1'])]
test = pd.read_csv('data/nCov_10k_test.csv')
sub = pd.read_csv('data/submit_example.csv')

train_content = train['微博中文内容'].values
train_label = train['情感倾向'].values.astype(int) + 1

test_content = test['微博中文内容'].values
test_label = test_content

oof_train = np.zeros((len(train), 3), dtype=np.float32)
oof_test = np.zeros((len(test), 3), dtype=np.float32)

model_path = 'F:/Study_documents/PycharmProjects/bert-base-chinese_pytorch/'

bert_config = BertConfig.from_pretrained(model_path + 'config.json', output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained(model_path + 'vocab.txt', config=bert_config)


def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
    softmax = x_exp / x_exp_row_sum
    return softmax


class BertForClass(nn.Module):
    def __init__(self, n_classes=3):
        super(BertForClass, self).__init__()
        self.model_name = 'BertForClass'
        self.bert_model = BertModel.from_pretrained(model_path, config=bert_config)
        self.classifier = nn.Linear(bert_config.hidden_size * 2, n_classes)

    def forward(self, input_ids, input_masks, segment_ids):
        sequence_output, pooler_output, hidden_states = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
                                                                        attention_mask=input_masks)
        seq_avg = torch.mean(sequence_output, dim=1) # dim = 1横向压缩
        concat_out = torch.cat((seq_avg, pooler_output), dim=1)
        logit = self.classifier(concat_out)

        return logit


class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='word_embeddings', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]


class data_generator:
    def __init__(self, data, batch_size=16, max_length=140, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.max_length = max_length
        self.shuffle = shuffle
        self.steps = len(self.data[0]) // self.batch_size
        if len(self.data[0]) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        c, y = self.data
        idxs = list(range(len(self.data[0])))
        if self.shuffle:
            np.random.shuffle(idxs)
        input_ids, input_masks, segment_ids, labels = [], [], [], []

        for index, i in enumerate(idxs):

            text = c[i]

            input_id = tokenizer.encode(text, max_length=self.max_length)
            input_mask = [1] * len(input_id)
            segment_id = [0] * len(input_id)
            padding_length = self.max_length - len(input_id)
            input_id += ([0] * padding_length)
            input_mask += ([0] * padding_length)
            segment_id += ([0] * padding_length)

            input_ids.append(input_id)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)
            labels.append(y[i])
            if len(input_ids) == self.batch_size or i == idxs[-1]:
                yield input_ids, input_masks, segment_ids, labels
                input_ids, input_masks, segment_ids, labels = [], [], [], []


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

epoch = 5

for fold, (train_index, valid_index) in enumerate(skf.split(train_content, train_label)):
    # if fold <= arg.cut:
    #    continue
    print('\n\n------------fold:{}------------\n'.format(fold))
    c = train_content[train_index]
    y = train_label[train_index]

    val_c = train_content[valid_index]
    val_y = train_label[valid_index]

    train_D = data_generator([c, y], batch_size=32, shuffle=True)
    val_D = data_generator([val_c, val_y], batch_size=32)

    model = BertForClass().to(device)
    pgd = PGD(model)
    K = 3
    loss_fn = nn.CrossEntropyLoss()

    optimizer = AdamW(params=model.parameters(), lr=1e-5)

    best_acc = 0
    PATH = './models/bert_{}.pth'.format(fold)
    for e in range(epoch):
        print('\n------------epoch:{}------------'.format(e))
        model.train()
        acc = 0
        train_len = 0
        loss_num = 0
        tq = tqdm(train_D)

        for input_ids, input_masks, segment_ids, labels in tq:
            input_ids = torch.tensor(input_ids).to(device)
            input_masks = torch.tensor(input_masks).to(device)
            segment_ids = torch.tensor(segment_ids).to(device)
            label_t = torch.tensor(labels, dtype=torch.long).to(device)

            y_pred = model(input_ids, input_masks, segment_ids)

            loss = loss_fn(y_pred, label_t)
            loss.backward()
            pgd.backup_grad()
            # 对抗训练
            for t in range(K):
                pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != K - 1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                y_pred = model(input_ids, input_masks, segment_ids)

                loss_adv = loss_fn(y_pred, label_t)
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            pgd.restore()  # 恢复embedding参数
            # 梯度下降，更新参数
            optimizer.step()
            model.zero_grad()

            y_max = np.argmax(y_pred.detach().to("cpu").numpy(), axis=1)  # 格式为[0]或[1]或[2]
            acc += sum(y_max == labels)
            loss_num += loss.item()
            train_len += len(labels)
            tq.set_postfix(fold=fold, epoch=e, loss=loss_num / train_len, acc=acc / train_len)

        model.eval()
        y_p = []
        for input_ids, input_masks, segment_ids, labels in tqdm(val_D):
            input_ids = torch.tensor(input_ids).to(device)
            input_masks = torch.tensor(input_masks).to(device)
            segment_ids = torch.tensor(segment_ids).to(device)
            label_t = torch.tensor(labels, dtype=torch.long).to(device)

            y_pred = model(input_ids, input_masks, segment_ids)

            y_max = np.argmax(y_pred.detach().to("cpu").numpy(), axis=1)
            y_p += list(y_max)

        acc = 0
        for i in range(len(y_p)):
            if val_y[i] == y_p[i]:
                acc += 1
        acc = acc / len(y_p)
        print("best_acc:{}  acc:{}\n".format(best_acc, acc))
        if acc >= best_acc:
            best_acc = acc
            torch.save(model, PATH)

    optimizer.zero_grad()

    model = torch.load(PATH).to(device)
    model.eval()
    y_p = []
    for input_ids, input_masks, segment_ids, labels in tqdm(val_D):
        input_ids = torch.tensor(input_ids).to(device)
        input_masks = torch.tensor(input_masks).to(device)
        segment_ids = torch.tensor(segment_ids).to(device)

        y_pred = model(input_ids, input_masks, segment_ids)

        y_p += y_pred.detach().to("cpu").tolist()

    y_p = np.array(y_p)

    oof_train[valid_index] = y_p

    test_D = data_generator([test_content, test_label], batch_size=32)
    model.eval()
    y_p = []
    for input_ids, input_masks, segment_ids, labels in tqdm(test_D):
        input_ids = torch.tensor(input_ids).to(device)
        input_masks = torch.tensor(input_masks).to(device)
        segment_ids = torch.tensor(segment_ids).to(device)

        y_pred = model(input_ids, input_masks, segment_ids)

        y_p += y_pred.detach().to("cpu").tolist()

    y_p = np.array(y_p)
    oof_test += y_p

np.save('./result/train.npy', oof_train)
np.save('./result/test_orgin.npy', oof_test)

x1 = np.array(oof_train)
from scipy import optimize


def fun(x):
    tmp = np.hstack([x[0] * x1[:, 0].reshape(-1, 1), x[1] * x1[:, 1].reshape(-1, 1), x[2] * x1[:, 2].reshape(-1, 1)])
    return -f1_score(train_label, np.argmax(tmp, axis=1), average='micro')


x0 = np.asarray((0, 0, 0))
res = optimize.fmin_powell(fun, x0)

xx_score = f1_score(train_label, np.argmax(oof_train, axis=1), average='micro')
print("原始f1_score", xx_score)
xx_cv = f1_score(train_label, np.argmax(oof_train * res, axis=1), average='micro')
print("修正后f1_score", xx_cv)

oof_test /= 5

oof_test_orgin = np.argmax(oof_test, axis=1)  # 返回list格式，如[0,1,2,2,0,1.....]
oof_test_optimizer = np.argmax(oof_test * res, axis=1)

for i in range(len(oof_test_orgin)):
    oof_test_orgin[i] -= 1

for i in range(len(oof_test_optimizer)):
    oof_test_optimizer[i] -= 1

np.save('./result/test_optimizer.npy', oof_test * res)

sub['y'] = oof_test_orgin
sub[['id', 'y']].to_csv('./result/bert_{}.csv'.format(xx_score), index=False)

sub['y'] = oof_test_optimizer
sub[['id', 'y']].to_csv('./result/bert_{}.csv'.format(xx_cv), index=False)
