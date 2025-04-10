import torch
import torch.nn.functional as F
from torch.nn import LayerNorm  # 使用 PyTorch 的 LayerNorm
from torch_geometric.nn import (
    GCNConv,
    GENConv,
    BatchNorm,
    global_mean_pool,
    global_max_pool,
    JumpingKnowledge,
    DeepGCNLayer,
)

class GCN(torch.nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=3, output_dim=1):
        """
        初始化模型的层次结构。

        参数：
        - input_dim (int): 输入节点特征的维度大小，默认为5。
        - hidden_dim (int): 隐藏层特征的维度大小。
        - num_layers (int): 深度图卷积层数，默认为3。
        - output_dim (int): 模型最终输出的维度，默认为1。
        """
        super().__init__()

        # 初始图卷积层：使用 GCNConv 将输入特征从 input_dim 映射到 hidden_dim
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)

        # 定义多个深度图卷积层，采用 GENConv，并使用残差连接
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = GENConv(
                hidden_dim, hidden_dim, aggr='softmax', t=1.0, learn_t=True, num_layers=2
            )  # 使用 GENConv 进行消息传递，num_layers 用于内部堆叠
            norm = LayerNorm(hidden_dim)  # 使用 LayerNorm 进行归一化
            act = torch.nn.PReLU(hidden_dim)  # 使用 PReLU 激活函数，提升非线性表达能力
            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.2)
            self.layers.append(layer)

        # 跳跃连接机制：采用“连接”（cat）模式，将初始层与每个深层的输出拼接
        self.jump = JumpingKnowledge(mode='cat')
        # 跳跃连接后，特征维度为 hidden_dim * (num_layers + 1)

        # 全局池化：结合均值池化和最大池化，拼接后维度为 2 * hidden_dim * (num_layers + 1)
        # 后续全连接网络用于进一步整合特征
        self.lin1 = torch.nn.Linear(hidden_dim * (num_layers + 1) * 2, hidden_dim * 2)
        self.lin2 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.lin_out = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        """
        前向传播函数，定义数据如何通过模型。

        参数：
        - data: 包含图数据的 Batch 对象，具有以下属性：
            - x (Tensor): 节点特征矩阵，形状为 [num_nodes, num_node_features]
            - edge_index (Tensor): 边的索引矩阵，形状为 [2, num_edges]
            - batch (Tensor): 每个节点所属图的索引，形状为 [num_nodes]

        返回：
        - x (Tensor): 模型的预测输出，形状为 [batch_size, output_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 初始图卷积层及 BatchNorm，ReLU 激活和 Dropout处理
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        # 保存初始层输出，便于后续跳跃连接
        xs = [x]

        # 逐层进行深度图卷积，并将每一层的输出保存
        for layer in self.layers:
            x = layer(x, edge_index)
            xs.append(x)

        # 跳跃连接：将所有层的输出进行拼接
        x = self.jump(xs)

        # 全局池化：同时采用均值池化和最大池化，充分捕捉图的全局信息
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # 全连接层：两次非线性映射和 Dropout，进一步整合特征后输出预测值
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin_out(x)

        return x