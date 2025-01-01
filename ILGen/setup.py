# setup.py
from setuptools import setup, find_packages

setup(
    name="ILGen",                     # 你的包名，可以自定义
    version="0.1.0",                  # 版本号
    packages=find_packages(),         # 自动查找含 __init__.py 的所有子目录
    
    # 如果你的三个函数或cli.py依赖特定第三方库，请在此列出
    install_requires=[
        "xgboost>=1.0.0",
        "pandas",
        "openpyxl",
        "rdkit",
    ],
    
    # 如果你需要把 xgboost_model/ 文件夹下的模型文件一并打包
    # 通常可以结合 include_package_data=True 使用
    include_package_data=True,

    # 如果你需要更精细地指定哪些文件被打包，可以使用 package_data 或者 MANIFEST.in
    package_data={
        # "ILGen" 对应包名（文件夹名）
        # 假设想把 ILGen/xgboost_model 下所有文件都打包进来
        "ILGen": ["xgboost_model/*"]
    },

    # 配置“命令行脚本”，将会在安装后自动生成一个名为 ILGen 的可执行命令
    entry_points={
        "console_scripts": [
            # 语法：命令名称=包.模块:函数
            # 这里表示在命令行里执行 "ILGen" 时，会运行 ILGen/cli.py 里的 main() 函数
            "ILGen=ILGen.cli:main"
        ]
    },

    # 简要的元数据，可选
    author="Jifeng Wang, Ying Wang",
    #author_email="YourEmail@example.com",
    description="A tool for generating ionic liquids",
    # url="https://github.com/YourRepo/ILGen",
    # ...
)