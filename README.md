# 硼氢化物储氢体系逆向设计工具
## 1. 项目概述
- **项目名称**: 硼氢化物储氢体系逆向设计工具
- **开发语言**: Python 3.11
- **开发环境**: 
  - 操作系统：Windows
  - IDE：PyCharm
  - 依赖库：`scikit-learn`，`pandas`，`numpy`，`matplotlib`，`pyside6`

## 2. 功能需求
该软件的核心功能包括：
- **数据输入**：
  - 用户可以输入硼氢化物的化学结构或者分子式.
    - 图1 常见的硼氢化物材料结构。
![结构图](https://github.com/user-attachments/assets/591870ba-9ef0-4ce7-8aae-f6f970047447)
    - 图2 元素周期表（其中绿色底面的是常见的与氢原子成键的元素，红色底面是镧系和锕系元素与氢原子组成氢键并不常见）
![元素周期表](https://github.com/user-attachments/assets/08c3934e-4007-420a-b9c4-21460e78560a)

- **特征提取**：
  - 从输入数据中提取化学特征，如电负性、键长、形成能等。
    - 图3 Magpie特征间皮尔逊相关系数图
 ![特性相关系数图](https://github.com/user-attachments/assets/920c060b-9d28-40a0-aa5e-ac6249affa88)

- **模型训练和预测**：
  - 使用已训练的机器学习模型（如线性回归模型、决策树模型、SVR模型等），预测与储氢相关的性质（如储氢能力、脱氢能量、最短氢键长度等）。
- **结果展示**：
  - 以图表或数据表格的形式展示预测结果。
    - 图4 XGBoost模型氢键长度预测结果与DFT计算值比较
![XGBoost模型预测拟合图](https://github.com/user-attachments/assets/0bade32e-56b3-44a6-be73-c59d83b91e98)

- **导入导出功能**：
  - 支持从文件（如CSV、Excel）导入化学结构数据，并将预测结果导出。

## 3. 系统架构
系统主要分为以下几个模块：
- **输入模块**：
  - 接收用户输入的化学结构信息，并进行格式化处理。
- **特征提取模块**：
  - 根据输入数据提取相关的分子特征。
- **机器学习模型模块**：
  - 通过预训练的模型（或在线训练）预测硼氢化物的性质。
- **结果展示模块**：
  - 通过图表或表格展示预测结果，并提供导出功能。

## 4. 后续开发计划
- 增加更多的硼氢化物性质预测功能
- 增加新功能，用户自己输入特征数据选择机器学习模型进行训练
- 增加更多的机器学习模型，提升不同性质的预测准确性
- 引入深度学习模型，提升预测准确性。
  
## 5. 部分代码展示
