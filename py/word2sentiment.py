#!/usr/bin/python
#-*-coding:utf-8-*-

import argparse
import collections
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import MeCab
tagger = MeCab.Tagger("-Owakati")
import time


parser = argparse.ArgumentParser()
parser.add_argument('--train', '-r', default='../data/ankABCD.txt', type=str)
parser.add_argument('--epoch', '-e', default=30, type=int, help='number of epochs')
parser.add_argument('--units', '-u', default=50, type=int, help='number of units per layer')
args = parser.parse_args()

batchsize = 25
EPOCH = args.epoch
n_units = args.units
ans_unit = 3


def ans3(ans):
    if ans < 2:
        return 0
    elif ans == 2:
        return 1
    else:
        return 2


def word2num(phrase, train):
    """
    単語を数字で置き換える
    未知の単語が入力された時0を返す
    出力はそれぞれ固有の単語を表す数字のlist(句)を返す
    """
    phrase_list = []
    for word in phrase:
        if word in vocab:
            pass
        elif train:
            vocab[word] = len(vocab) + 1
            phrase_list.append(vocab[word])
        else:
            vocab[word] = 0
        phrase_list.append(vocab[word])
    return phrase_list


def make_corpus(train):
    corpus = []
    for line in open(args.train):
        # time,C,10月27日,14:18,動物は好きですか？,はい,4
        item = line.strip().split(',')
        wakati = tagger.parse(item[5]).split()
        ans = int(item[6]) - 1
        ans = ans3(ans)
        wakati_num = word2num(wakati, train)
        corpus.append((wakati_num, ans))
    return corpus


# Neural net architecture
# ニューラルネットの構造
def forward(x_data, y_data, train=True):
    w1 = np.array([x_data[0]], np.int32)
    x1 = chainer.Variable(w1, volatile=not train)
    p1 = model.embed(x1)
    for word in x_data[1:]:
        w2 = np.array([x_data[0]], np.int32)
        x2 = chainer.Variable(w2, volatile=not train)
        p2 = model.embed(x2)
        #print p1.data
        #単語をsum
        p1 = p1 + p2
    #print p1.data
    h2 = F.dropout(F.relu(model.l1(p1)), train=train)
    y = model.l2(h2)
    #正解データをVariableに変換
    y_data_v = np.array([y_data], np.int32)
    t = chainer.Variable(y_data_v)

    # 多クラス分類なので誤差関数としてソフトマックス関数の
    # 交差エントロピー関数を用いて、誤差を導出
    loss = F.softmax_cross_entropy(y, t)
    predict = cuda.to_cpu(y.data).argmax(1)
    if predict[0] == ans:
        result['correct_node'] += 1
        result['correct_' + str(ans)] += 1
    result['total_node'] += 1
    result['total_' + str(ans)] += 1
    result['pred_' + str(predict[0])] += 1
    return loss

# コーパス定義
vocab = {}
all_corpus = make_corpus(train=True)
np.random.shuffle(all_corpus)
l = len(all_corpus)
train_corpus = all_corpus[: int(l * 0.8)]
test_corpus = all_corpus[int(l * 0.8):]

# Prepare multi-layer perceptron model
# 多層パーセプトロンモデルの設定
# 入力w2v? 中間50次元、出力 5次元
model = chainer.FunctionSet(
    embed=F.EmbedID(len(vocab) + 1, n_units),
    l1=F.Linear(n_units, n_units),
    l2=F.Linear(n_units, ans_unit)
)
# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model.collect_parameters())


start_at = time.time()
accum_loss, count = 0, 0
for epoch in range(EPOCH):
    result = collections.defaultdict(lambda: 0)
    print('-------- Epoch: {0:d} --------'.format(epoch))
    total_loss = 0
    cur_at = time.time()
    np.random.shuffle(train_corpus)
    for words, ans in train_corpus:
        # 順伝播させて誤差と精度を算出
        loss = forward(words, ans, train=True)
        accum_loss += loss
        count += 1
        if count >= batchsize:
            optimizer.zero_grads()
            #逆伝播
            accum_loss.backward()
            optimizer.weight_decay(0.0001)
            optimizer.update()
            total_loss += float(cuda.to_cpu(accum_loss.data))
            accum_loss = 0
            count = 0
    print('all loss: {:.4f}'.format(total_loss))
    now = time.time()
    print('{:.2f} sec'.format(now - cur_at))
    print('Train data evaluation:')
    acc_node = 100.0 * result['correct_node'] / result['total_node']
    print('Accuracy: {0:.2f}  %%  ({1:,d}/{2:,d})'.format(
        acc_node, result['correct_node'], result['total_node']))
    print result
