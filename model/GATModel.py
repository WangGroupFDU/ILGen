import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, BatchNorm

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, BatchNorm

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, num_heads=4):
        """
        初始化函数，定义模型的层次结构。

        参数：
        - in_channels (int): 输入节点特征的维度，即每个节点特征向量的长度。
        - hidden_channels (int): 隐藏层特征维度。
        - out_channels (int): 输出特征维度（预测目标的维度）。
        - num_heads (int): 多头注意力的头数。

        例如：
        如果输入节点特征维度为11，隐藏层维度为8，输出维度为1，注意力头数为4：
        model = GAT(in_channels=11, hidden_channels=8, out_channels=1, num_heads=4)
        """
        super(GAT, self).__init__()
        # 第1层 GATConv
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)
        # 第2层 GATConv
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True)
        # 第3层 GATConv
        self.conv3 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, concat=False)
        # 将输出特征映射为 1 维
        self.fc = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        """
        前向传播函数，定义数据如何通过模型。
        
        参数：
        - data: 包含图数据的Batch对象，具有以下属性：
            - x (Tensor): 节点特征矩阵，形状为 [num_nodes, in_channels]
            - edge_index (Tensor): 边的索引矩阵，形状为 [2, num_edges]
            - batch (Tensor): 每个节点所属的图的索引，形状为 [num_nodes]

        返回：
        - out (Tensor): 模型的最终预测输出，形状为 [batch_size, out_channels]
        - x_128 (Tensor): 经过全连接层映射后的 128 维向量（读出层），形状为 [batch_size, 128]
        """
        x = data.x  # 节点特征矩阵，形状: [num_nodes, in_channels]
        edge_index = data.edge_index  # 边索引矩阵，形状: [2, num_edges]
        batch = data.batch  # 节点所属的图的索引，形状: [num_nodes]

        # 第一次 GATConv
        x = self.conv1(x, edge_index)
        x = F.elu(x)  # 激活函数

        # 第二次 GATConv
        x = self.conv2(x, edge_index)
        x = F.elu(x)

        # 第三次 GATConv
        x = self.conv3(x, edge_index)
        x = F.elu(x)

        # 全局平均池化，将节点特征聚合为图的特征
        x = global_mean_pool(x, batch)
        # x 形状更新为 [batch_size, hidden_channels]

        # Dropout 防止过拟合
        x = F.dropout(x, p=0.5, training=self.training)
        # 形状仍为 [batch_size, hidden_channels]

        # 将输出映射为1维
        x = self.fc(x)

        return x