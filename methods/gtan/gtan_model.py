import torch
import torch.nn as nn
import torch.optim as optim
from dgl.utils import expand_as_pair
from dgl import function as fn
from dgl.base import DGLError
from dgl.nn.functional import edge_softmax
import numpy as np
cat_features = ["Target",
                "Type",
                "Location"]


class PosEncoding(nn.Module):
#生成的位置编码是基于正弦函数的，它可以为每个位置索引生成一个唯一的编码向量。
# 这种编码方式在处理图数据时特别有用，因为它可以捕捉节点之间的相对或绝对位置关系。
    def __init__(self, dim, device, base=10000, bias=0):

        super(PosEncoding, self).__init__()
        """
        Initialize the posencoding component
        :param dim: the encoding dimension 
		:param device: where to train model
		:param base: the encoding base编码的底数
		:param bias: the encoding bias
        """
        p = [] #存储每个维度的底数幂次，用于位置编码
        sft = []#存储正弦函数的相位偏移，用于确保正负位置的对称性
        for i in range(dim):
            b = (i - i % 2) / dim
            p.append(base ** -b)
            if i % 2:
                sft.append(np.pi / 2.0 + bias)
            else:
                sft.append(bias)
        self.device = device
        self.sft = torch.tensor(
            sft, dtype=torch.float32).view(1, -1).to(device)
        self.base = torch.tensor(p, dtype=torch.float32).view(1, -1).to(device)

    def forward(self, pos):
        with torch.no_grad():
            if isinstance(pos, list):
                pos = torch.tensor(pos, dtype=torch.float32).to(self.device)
            pos = pos.view(-1, 1)
            x = pos / self.base + self.sft # 计算位置编码
            return torch.sin(x)


class TransEmbedding(nn.Module):

    def __init__(self, df=None, device='cpu', dropout=0.2, in_feats=82, cat_features=None):
        """
        Initialize the attribute embedding and feature learning compoent

        :param df: the feature
                :param device: where to train model
                :param dropout: the dropout rate
                :param in_feat: the shape of input feature in dimension 1
                :param cat_feature: category features
        """
        super(TransEmbedding, self).__init__()
        self.time_pe = PosEncoding(dim=in_feats, device=device, base=100)
        #time_emb = time_pe(torch.sin(torch.tensor(df['time_span'].values)/86400*torch.pi))
        # 创建一个包含多个嵌入层的字典 self.cat_table。每个嵌入层对应于一个类别特征的嵌入。
        self.cat_table = nn.ModuleDict({col: nn.Embedding(max(df[col].unique(
        ))+1, in_feats).to(device) for col in cat_features if col not in {"Labels", "Time"}})
        self.label_table = nn.Embedding(3, in_feats, padding_idx=2).to(device)
        self.time_emb = None
        self.emb_dict = None
        self.label_emb = None
        self.cat_features = cat_features
        # 创建一个前向传播的MLP列表 forward_mlp，用于处理每个类别特征的嵌入
        self.forward_mlp = nn.ModuleList(
            [nn.Linear(in_feats, in_feats) for i in range(len(cat_features))])
        self.dropout = nn.Dropout(dropout)

    def forward_emb(self, df):
        # 生成类别特征的嵌入
        if self.emb_dict is None:
            self.emb_dict = self.cat_table
        # print(self.emb_dict)
        # print(df['trans_md'])
        # 遍历 cat_features 列表，对于每个特征列（除了 "Labels" 和 "Time"），使用相应的嵌入层生成嵌入
        # self.emb_dict[col](df[col]) 是调用嵌入层，将类别特征的值转换为嵌入向量
        # df[col]已经转换成onehot编码了吗？在哪实现的？？
        support = {col: self.emb_dict[col](
            df[col]) for col in self.cat_features if col not in {"Labels", "Time"}}
        #self.time_emb = self.time_pe(torch.sin(torch.tensor(df['time_span'])/86400*torch.pi))
        #support['time_span'] = self.time_emb
        #support['labels'] = self.label_table(df['labels'])
        return support

    def forward(self, df):
        support = self.forward_emb(df)
        output = 0
        for i, k in enumerate(support.keys()):
            # if k =='time_span':
            #    print(df[k].shape)
            support[k] = self.dropout(support[k])
            support[k] = self.forward_mlp[i](support[k])
            output = output + support[k]
        return output


class TransformerConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 bias=True,
                 allow_zero_in_degree=False,
                 # feat_drop=0.6,
                 # attn_drop=0.6,
                 skip_feat=True,
                 gated=True,
                 layer_norm=True,
                 activation=nn.PReLU()):
        """
        Initialize the transformer layer.
        Attentional weights are jointly optimized in an end-to-end mechanism with graph neural networks and fraud detection networks.
            :param in_feat: the shape of input feature
            :param out_feats: the shape of output feature
            :param num_heads: the number of multi-head attention 
            :param bias: whether to use bias
            :param allow_zero_in_degree: whether to allow zero in degree
            :param skip_feat: whether to skip some feature 
            :param gated: whether to use gate
            :param layer_norm: whether to use layer regularization
            :param activation: the type of activation function   
        """

        super(TransformerConv, self).__init__()
        # expand_as_pair接受一个参数 in_feats 并将其扩展为一对值，即源节点和目标节点的输入特征维度
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._num_heads = num_heads

        # 创建线性层 lin_query、lin_key 和 lin_value，用于生成查询（Q）、键（K）和值（V）的表示
        self.lin_query = nn.Linear(
            self._in_src_feats, self._out_feats*self._num_heads, bias=bias)
        self.lin_key = nn.Linear(
            self._in_src_feats, self._out_feats*self._num_heads, bias=bias)
        self.lin_value = nn.Linear(
            self._in_src_feats, self._out_feats*self._num_heads, bias=bias)

        #self.feat_dropout = nn.Dropout(p=feat_drop)
        #self.attn_dropout = nn.Dropout(p=attn_drop)
        # 如果 skip_feat 为 True，则创建一个跳过连接的线性层 skip_feat
        if skip_feat:
            self.skip_feat = nn.Linear(
                self._in_src_feats, self._out_feats*self._num_heads, bias=bias)
        else:
            self.skip_feat = None
        # 如果 gated 为 True，则创建一个门控线性层 gate
        if gated:
            self.gate = nn.Linear(
                3*self._out_feats*self._num_heads, 1, bias=bias)
        else:
            self.gate = None
        # 如果 layer_norm 为 True，则创建一个层归一化 layer_norm
        if layer_norm:
            self.layer_norm = nn.LayerNorm(self._out_feats*self._num_heads)
        else:
            self.layer_norm = None
        self.activation = activation

    def forward(self, graph, feat, get_attention=False):
        """
        Description: Transformer Graph Convolution
        :param graph: input graph
            :param feat: input feat
            :param get_attention: whether to get attention是否返回注意力分数
        """

        graph = graph.local_var()

        if not self._allow_zero_in_degree:
            if (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. '
                               'This is harmful for some applications, '
                               'causing silent performance regression. '
                               'Adding self-loop on the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve '
                               'the issue. Setting ``allow_zero_in_degree`` '
                               'to be `True` when constructing this module will '
                               'suppress the check and let the code run.')

        # check if feat is a tuple
        if isinstance(feat, tuple):
            # 分别获取源节点和目标节点的特征
            h_src = feat[0]
            h_dst = feat[1]
        else:
            h_src = feat
            h_dst = h_src[:graph.number_of_dst_nodes()]

        # Step 0. q, k, v
        # 使用线性层生成 Q、K 和 V 的表示，并将它们重塑为多头注意力的形式
        q_src = self.lin_query(
            h_src).view(-1, self._num_heads, self._out_feats)
        k_dst = self.lin_key(h_dst).view(-1, self._num_heads, self._out_feats)
        v_src = self.lin_value(
            h_src).view(-1, self._num_heads, self._out_feats)
        # Assign features to nodes
        # 将 Q 和 V 的特征分配给图的源节点，将 K 的特征分配给目标节点
        graph.srcdata.update({'ft': q_src, 'ft_v': v_src})
        graph.dstdata.update({'ft': k_dst})
        # Step 1. dot product
        # 计算 Q 和 K 的点积，得到注意力分数
        graph.apply_edges(fn.u_dot_v('ft', 'ft', 'a'))

        # Step 2. edge softmax to compute attention scores
        # 对注意力分数进行 softmax 操作，得到标准化的注意力分数
        graph.edata['sa'] = edge_softmax(
            graph, graph.edata['a'] / self._out_feats**0.5)

        # Step 3. Broadcast softmax value to each edge, and aggregate dst node
        # 使用注意力分数聚合 V 的特征，得到目标节点的更新特征
        graph.update_all(fn.u_mul_e('ft_v', 'sa', 'attn'),
                         fn.sum('attn', 'agg_u'))

        # output results to the destination nodes
        rst = graph.dstdata['agg_u'].reshape(-1,
                                             self._out_feats*self._num_heads)

        # 如果存在跳过连接，则将跳过连接的特征与聚合后的特征结合
        if self.skip_feat is not None:
            # feat[:graph.number_of_dst_nodes()] 选择了输入特征 feat 中对应于图中目标节点数量的部分。
            # 这是因为在图神经网络中，跳过连接的特征通常与目标节点相关联
            # 跳跃连接特征通常包含了原始输入特征或前一层的输出特征
            skip_feat = self.skip_feat(feat[:graph.number_of_dst_nodes()])
            # 如果存在门控机制，则使用门控线性层计算门控信号，并结合跳过连接和聚合后的特征
            if self.gate is not None:
                gate = torch.sigmoid(
                    self.gate(#将三个特征向量（原始跳过连接特征、当前层的输出特征和它们之间的差异）在最后一个维度（dim=-1）上进行拼接
                        torch.concat([skip_feat, rst, skip_feat - rst], dim=-1)))
                rst = gate * skip_feat + (1 - gate) * rst
            else:
                rst = skip_feat + rst

        if self.layer_norm is not None:
            rst = self.layer_norm(rst)

        if self.activation is not None:
            rst = self.activation(rst)

        if get_attention:
            return rst, graph.edata['sa']
        else:
            return rst


class GraphAttnModel(nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 heads,
                 activation,
                 skip_feat=True,
                 gated=True,
                 layer_norm=True,
                 post_proc=True,
                 n2v_feat=True,
                 drop=None,
                 ref_df=None,
                 cat_features=None,
                 nei_features=None,
                 device='cpu'):
        """
        Initialize the GTAN-GNN model
        :param in_feats: the shape of input feature
                :param hidden_dim: model hidden layer dimension
                :param n_layers: the number of GTAN layers
                :param n_classes: the number of classification
                :param heads: the number of multi-head attention 
                :param activation: the type of activation function
                :param skip_feat: whether to skip some feature
                :param gated: whether to use gate
        :param layer_norm: whether to use layer regularization
                :param post_proc: whether to use post processing
                :param n2v_feat: whether to use n2v features
        :param drop: whether to use drop
                :param ref_df: whether to refer other node features
                :param cat_features: category features
                :param nei_features: neighborhood statistic features
        :param device: where to train model
        """

        super(GraphAttnModel, self).__init__()
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.heads = heads
        self.activation = activation
        #self.input_drop = lambda x: x
        self.input_drop = nn.Dropout(drop[0])
        self.drop = drop[1]
        self.output_drop = nn.Dropout(self.drop)
        # self.pn = PairNorm(mode=pairnorm)
        if n2v_feat:
            self.n2v_mlp = TransEmbedding(
                ref_df, device=device, in_feats=in_feats, cat_features=cat_features)
        else:
            self.n2v_mlp = lambda x: x
        # 添加线性层，用于特征转换
        self.layers = nn.ModuleList()
        self.layers.append(nn.Embedding(
            n_classes+1, in_feats, padding_idx=n_classes))
        self.layers.append(
            nn.Linear(self.in_feats, self.hidden_dim*self.heads[0]))
        self.layers.append(
            nn.Linear(self.in_feats, self.hidden_dim*self.heads[0]))
        self.layers.append(nn.Sequential(nn.BatchNorm1d(self.hidden_dim*self.heads[0]),
                                         nn.PReLU(),
                                         nn.Dropout(self.drop),
                                         nn.Linear(self.hidden_dim *
                                                   self.heads[0], in_feats)
                                         ))

        # build multiple layers
        self.layers.append(TransformerConv(in_feats=self.in_feats,
                                           out_feats=self.hidden_dim,
                                           num_heads=self.heads[0],
                                           skip_feat=skip_feat,
                                           gated=gated,
                                           layer_norm=layer_norm,
                                           activation=self.activation))

        for l in range(0, (self.n_layers - 1)):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(TransformerConv(in_feats=self.hidden_dim * self.heads[l - 1],
                                               out_feats=self.hidden_dim,
                                               num_heads=self.heads[l],
                                               skip_feat=skip_feat,
                                               gated=gated,
                                               layer_norm=layer_norm,
                                               activation=self.activation))
        if post_proc:
            self.layers.append(nn.Sequential(nn.Linear(self.hidden_dim * self.heads[-1], self.hidden_dim * self.heads[-1]),
                                             nn.BatchNorm1d(
                                                 self.hidden_dim * self.heads[-1]),
                                             nn.PReLU(),
                                             nn.Dropout(self.drop),
                                             nn.Linear(self.hidden_dim * self.heads[-1], self.n_classes)))
        else:
            self.layers.append(nn.Linear(self.hidden_dim *
                               self.heads[-1], self.n_classes))

    def forward(self, blocks, features, labels, n2v_feat=None):
        """
        :param blocks: train blocks
        :param features: train features  (|input|, feta_dim)
        :param labels: train labels (|input|, )
        :param n2v_feat: whether to use n2v features
        从gtan_main传来的分别为blocks、数值特征、修改后的标签、类别特征（不含label）
        """

        if n2v_feat is None:
            h = features
        else:
            h = self.n2v_mlp(n2v_feat)
            h = features + h

        # 使用嵌入层和线性层处理标签，得到标签嵌入
        label_embed = self.input_drop(self.layers[0](labels))
        label_embed = self.layers[1](h) + self.layers[2](label_embed)
        label_embed = self.layers[3](label_embed)
        h = h + label_embed  # 将标签嵌入与特征进行残差连接

        #遍历所有 GTAN 层，将特征通过每个层进行前向传播
        for l in range(self.n_layers):
            h = self.output_drop(self.layers[l+4](blocks[l], h))

        # 使用最后一个层将特征转换为预测的 logits
        logits = self.layers[-1](h)

        return logits
