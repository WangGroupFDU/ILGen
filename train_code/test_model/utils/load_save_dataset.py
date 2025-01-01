import torch
# 加载数据集
def load_data_list(file_path):
    """
    从指定文件路径加载数据列表。

    参数：
    file_path (str): 数据列表文件的路径。

    返回：
    list: 加载的数据列表。
    """
    return torch.load(file_path)

def save_data_list(data_list, file_path):
    """
    保存数据列表到指定文件路径。

    参数：
    data_list (list): 需要保存的数据列表。
    file_path (str): 保存文件的路径。
    """
    torch.save(data_list, file_path)

