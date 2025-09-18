# 🔮 SparseTSF: PyTorch 复现与 MindSpore 迁移

本项目基于 **时间序列预测模型 SparseTSF**，我们完成了在 **PyTorch** 框架上的复现，并将代码迁移到 **MindSpore** 框架，确保能在相同数据集上运行与复现结果。

---

## 🚀 项目结构
```
SparseTSF/
├── pytorch_version/ # PyTorch 实现与实验代码
├── mindspore_version/ # MindSpore 迁移版本
├── data/ # 数据集存放目录
├── results/ # 实验结果与日志
├── requirements.txt
└── README.md
```
---

## ⚙️ 本地运行方式

### 1. 安装依赖
```bash
pip install -r requirements.txt
```
### 2. 运行 PyTorch 版本
```bash
cd pytorch_version
python main.py --dataset ETTh1
```
### 3. 运行 MindSpore 版本
```bash
cd mindspore_version
python main.py --dataset ETTh1
```
---

## 🗄️ 数据集说明

常用数据集包括：

ETTh1

ETTh2

Electricity

Traffic

将下载的数据集放置于 data/ 目录下。

---

## 📊 实验结果

PyTorch 与 MindSpore 两个版本均可在主流长序列预测数据集上稳定运行。

---

