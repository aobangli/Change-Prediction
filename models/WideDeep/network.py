import torch
import torch.nn as nn


class Wide(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Wide, self).__init__()
        # hand-crafted cross-product features
        self.linear = nn.Linear(in_features=input_dim, out_features=output_dim)

    def forward(self, x):
        return self.linear(x)


class Deep(nn.Module):
    def __init__(self, deep_dropout, hidden_layers):
        super(Deep, self).__init__()
        self.dnn = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_layers[:-1], hidden_layers[1:]))])
        self.dropout = nn.Dropout(p=deep_dropout)

    def forward(self, x):
        for layer in self.dnn:
            x = layer(x)
            # 如果输出层大小是1的话，这里再使用了个ReLU激活函数，可能导致输出全变成0，即造成了梯度消失，导致Loss不收敛
            x = torch.relu(x)
        x = self.dropout(x)
        return x


class WideDeep(nn.Module):
    def __init__(self, num_of_dense_feature, sparse_features_val_num, deep_dropout, embed_dim, hidden_layers, output_dim):
        super(WideDeep, self).__init__()

        # 稠密特征的数量
        self._num_of_dense_feature = num_of_dense_feature
        # 稀疏特征类别数
        self.sparse_features_val_num = sparse_features_val_num

        self.embedding_layers = nn.ModuleList([
            # 根据稀疏特征的个数创建对应个数的Embedding层，Embedding输入大小是稀疏特征的类别总数，输出稠密向量的维度由config文件配置
            nn.Embedding(num_embeddings=num_feat, embedding_dim=embed_dim)
            for num_feat in self.sparse_features_val_num
        ])

        # Deep hidden layers
        self._deep_hidden_layers = hidden_layers
        self._deep_hidden_layers.insert(0, self._num_of_dense_feature + embed_dim * len(
            self.sparse_features_val_num))

        self._wide = Wide(self._num_of_dense_feature, output_dim)
        self._deep = Deep(deep_dropout, self._deep_hidden_layers)
        # 之前直接将这个final_layer加入到了Deep模块里面，想着反正输出都是1，结果没注意到Deep没经过一个Linear层都会经过Relu激活函数，如果
        # 最后输出层大小是1的话，再经过ReLU之后，很可能变为了0，造成梯度消失问题，导致Loss怎么样都降不下来。
        self._final_linear = nn.Linear(self._deep_hidden_layers[-1], output_dim)

    def forward(self, x):
        # 先区分出稀疏特征和稠密特征，这里是按照列来划分的，即所有的行都要进行筛选
        # 不能用下面这行区分，此时的数据x已经没有列名信息，只能通过列数来取，靠后的列是稀疏特征
        # dense_input, sparse_inputs = x[:, self.dense_features_cols], x[:, self.sparse_features_cols]
        dense_input, sparse_inputs = x[:, :self._num_of_dense_feature], x[:, self._num_of_dense_feature:]
        sparse_inputs = sparse_inputs.long()

        sparse_embeds = [self.embedding_layers[i](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        sparse_embeds = torch.cat(sparse_embeds, dim=-1)
        # Deep模块的输入是稠密特征和稀疏特征经过Embedding产生的稠密特征的
        deep_input = torch.cat([sparse_embeds, dense_input], dim=-1)

        wide_out = self._wide(dense_input)
        deep_out = self._deep(deep_input)
        deep_out = self._final_linear(deep_out)

        assert (wide_out.shape == deep_out.shape)

        outputs = torch.sigmoid(0.5 * (wide_out + deep_out))
        return outputs

    # def saveModel(self):
    #     torch.save(self.state_dict(), self._config['model_name'])
    #
    # def loadModel(self, map_location):
    #     state_dict = torch.load(self._config['model_name'], map_location=map_location)
    #     self.load_state_dict(state_dict, strict=False)
