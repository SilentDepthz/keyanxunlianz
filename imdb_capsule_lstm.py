import logging
import os
import sys
import pickle
import time

import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
from tqdm import tqdm

from sklearn.metrics import accuracy_score

test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)

num_epochs = 10
embed_size = 300
num_hiddens = 128
num_layers = 2
bidirectional = True
batch_size = 64
labels = 2
lr = 0.01
device = torch.device('cuda:0')
use_gpu = True


class Capsule(nn.Module):
    def __init__(self, num_hiddens, bidirectional, num_capsule=5, dim_capsule=5, routings=4, **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.bidirectional = bidirectional

        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.activation = self.squash

        # 修复权重形状计算
        if self.bidirectional:
            input_dim = self.num_hiddens * 2
            output_dim = self.num_capsule * self.dim_capsule
            self.W = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(1, input_dim, output_dim)))
        else:
            input_dim = self.num_hiddens
            output_dim = self.num_capsule * self.dim_capsule
            self.W = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(1, input_dim, output_dim)))

    def forward(self, inputs):
        # inputs 形状: [seq_len, batch_size, hidden_size*2]
        # 我们需要转换为 [batch_size, seq_len, hidden_size*2]
        inputs = inputs.permute(1, 0, 2)  # [batch_size, seq_len, hidden_size*2]

        batch_size, seq_len, hidden_dim = inputs.shape

        # 矩阵乘法: [batch_size, seq_len, hidden_dim] * [1, hidden_dim, num_capsule*dim_capsule]
        u_hat_vecs = torch.matmul(inputs, self.W)  # [batch_size, seq_len, num_capsule*dim_capsule]

        # 重塑为胶囊格式
        u_hat_vecs = u_hat_vecs.view(batch_size, seq_len, self.num_capsule, self.dim_capsule)

        # 调整维度顺序: [batch_size, num_capsule, seq_len, dim_capsule]
        u_hat_vecs = u_hat_vecs.permute(0, 2, 1, 3).contiguous()

        # 动态路由
        with torch.no_grad():
            b = torch.zeros(batch_size, self.num_capsule, seq_len, 1).to(inputs.device)

        for i in range(self.routings):
            c = F.softmax(b, dim=1)  # [batch_size, num_capsule, seq_len, 1]
            outputs = self.activation(torch.sum(c * u_hat_vecs, dim=2))  # [batch_size, num_capsule, dim_capsule]

            if i < self.routings - 1:
                # 更新路由系数
                outputs_expanded = outputs.unsqueeze(2)  # [batch_size, num_capsule, 1, dim_capsule]
                update = torch.sum(outputs_expanded * u_hat_vecs, dim=-1,
                                   keepdim=True)  # [batch_size, num_capsule, seq_len, 1]
                b = b + update

        return outputs  # [batch_size, num_capsule, dim_capsule]

    @staticmethod
    def squash(x, axis=-1):
        s_squared_norm = (x ** 2).sum(axis, keepdim=True)
        scale = torch.sqrt(s_squared_norm + 1e-7)
        return x / scale


class SentimentNet(nn.Module):
    def __init__(self, embed_size, num_hiddens, num_layers, bidirectional, weight, labels, use_gpu, **kwargs):
        super(SentimentNet, self).__init__(**kwargs)
        self.embed_size = embed_size
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        self.encoder = nn.LSTM(input_size=self.embed_size, hidden_size=self.num_hiddens,
                               num_layers=self.num_layers, bidirectional=self.bidirectional,
                               dropout=0)
        self.capsule = Capsule(num_hiddens=self.num_hiddens, bidirectional=self.bidirectional)

        # 修复解码器输入维度
        # 胶囊层输出: [batch_size, num_capsule, dim_capsule] -> 展平后: [batch_size, num_capsule * dim_capsule]
        capsule_output_dim = 5 * 5  # num_capsule * dim_capsule
        self.decoder = nn.Linear(capsule_output_dim, labels)

    def forward(self, inputs):
        # 嵌入层
        embeddings = self.embedding(inputs)  # [batch_size, seq_len, embed_size]

        # LSTM层
        states, hidden = self.encoder(embeddings.permute(1, 0, 2))  # states: [seq_len, batch_size, hidden_size*2]

        # 胶囊层
        capsule_output = self.capsule(states)  # [batch_size, num_capsule, dim_capsule]

        # 展平胶囊输出
        batch_size = capsule_output.size(0)
        capsule_flat = capsule_output.view(batch_size, -1)  # [batch_size, num_capsule * dim_capsule]

        # 分类层
        outputs = self.decoder(capsule_flat)

        return outputs


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    pickle_file = os.path.join('pickle', 'F:\keyanxunlian\imdb_sentiment_analysis_torch\pickle\imdb_glove.pickle3')
    [train_features, train_labels, val_features, val_labels, test_features, weight, word_to_idx, idx_to_word,
     vocab] = pickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')

    # 先测试一个小批量以确保形状正确
    print("测试网络形状...")
    net = SentimentNet(embed_size=embed_size, num_hiddens=num_hiddens, num_layers=num_layers,
                       bidirectional=bidirectional, weight=weight,
                       labels=labels, use_gpu=use_gpu)
    net.to(device)

    # 测试形状
    with torch.no_grad():
        test_input = train_features[:2].to(device)  # 用2个样本测试
        print(f"测试输入形状: {test_input.shape}")
        test_output = net(test_input)
        print(f"测试输出形状: {test_output.shape}")
        print("形状测试通过!")

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    val_set = torch.utils.data.TensorDataset(val_features, val_labels)
    test_set = torch.utils.data.TensorDataset(test_features, )

    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        start = time.time()
        train_loss, val_losses = 0, 0
        train_acc, val_acc = 0, 0
        n, m = 0, 0
        with tqdm(total=len(train_iter), desc='Epoch %d' % epoch) as pbar:
            for feature, label in train_iter:
                n += 1
                net.zero_grad()
                feature = Variable(feature.cuda())
                label = Variable(label.cuda())
                score = net(feature)
                loss = loss_function(score, label)
                loss.backward()
                optimizer.step()
                train_acc += accuracy_score(torch.argmax(score.cpu().data,
                                                         dim=1), label.cpu())
                train_loss += loss

                pbar.set_postfix({'epoch': '%d' % (epoch),
                                  'train loss': '%.4f' % (train_loss.data / n),
                                  'train acc': '%.2f' % (train_acc / n)
                                  })
                pbar.update(1)

            with torch.no_grad():
                for val_feature, val_label in val_iter:
                    m += 1
                    val_feature = val_feature.cuda()
                    val_label = val_label.cuda()
                    val_score = net(val_feature)
                    val_loss = loss_function(val_score, val_label)
                    val_acc += accuracy_score(torch.argmax(val_score.cpu().data, dim=1), val_label.cpu())
                    val_losses += val_loss
            end = time.time()
            runtime = end - start
            pbar.set_postfix({'epoch': '%d' % (epoch),
                              'train loss': '%.4f' % (train_loss.data / n),
                              'train acc': '%.2f' % (train_acc / n),
                              'val loss': '%.4f' % (val_losses.data / m),
                              'val acc': '%.2f' % (val_acc / m),
                              'time': '%.2f' % (runtime)
                              })

    test_pred = []
    with torch.no_grad():
        with tqdm(total=len(test_iter), desc='Prediction') as pbar:
            for test_feature, in test_iter:
                test_feature = test_feature.cuda()
                test_score = net(test_feature)
                test_pred.extend(torch.argmax(test_score.cpu().data, dim=1).numpy().tolist())
                pbar.update(1)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("./result/capsule_lstm.csv", index=False, quoting=3)
    logging.info('result saved!')