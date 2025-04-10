import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.3):
        """
        初始化函数，定义MLP的层次结构，同时引入BatchNorm和Dropout以防止过拟合。

        参数：
        - input_size (int): 输入特征的维度，即每个样本特征向量的长度。
        - hidden_sizes (list of int): 隐藏层的神经元数量列表。
        - output_size (int): 输出层的神经元数量，通常对应于分类任务的类别数或回归任务的目标维度。
        - dropout_rate (float): Dropout比例，用于抑制过拟合，默认值为0.3。
        
        示例：
        如果输入特征维度为20，隐藏层神经元数量为[64, 32]，输出维度为10：
        model = MLP(input_size=20, hidden_sizes=[64, 32], output_size=10, dropout_rate=0.3)
        """
        super(MLP, self).__init__()
        layers = []
        in_dim = input_size  # 当前层输入维度初始为input_size
        
        # 构建隐藏层：每层包含线性映射、BatchNorm、ReLU激活和Dropout
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_size))            # 线性层
            layers.append(nn.BatchNorm1d(hidden_size))                 # 批归一化层，有助于稳定训练
            layers.append(nn.ReLU())                                   # ReLU激活函数，增加非线性
            layers.append(nn.Dropout(dropout_rate))                    # Dropout层，防止过拟合
            in_dim = hidden_size                                      # 更新下一层的输入维度
        
        # 添加输出层，不使用激活函数（可根据任务需要在损失函数中处理）
        layers.append(nn.Linear(in_dim, output_size))
        
        # 将所有层组合成一个顺序模型
        self.network = nn.Sequential(*layers)

    def forward(self, data):
        """
        前向传播函数，定义数据如何通过模型。

        参数：
        - data: 包含输入数据的对象，其中 data.morgan_fp 是分子指纹，形状为 [batch_size, input_size]

        返回：
        - out (Tensor): 模型的输出，形状为 [batch_size, output_size]
        """
        x = data.morgan_fp  # 以分子指纹作为输入
        out = self.network(x)  # 数据通过网络进行前向传播
        return out