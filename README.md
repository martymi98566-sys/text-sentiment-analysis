# 电影评论情绪分析与原因可视化

本项目基于 [IvoryGate/text-sentiment-analysis](https://github.com/IvoryGate/text-sentiment-analysis) 延续开发，保留原项目提交历史、模型对照实验与目录结构。当前版本增加本地交互网页：输入一条英文影评，展示整体积极或消极、原文情绪片段高亮，以及“原因大类 × 好坏”的加权占比。

**当前版本为研究试用版 `v0.2.0-research`。** 原因分类的新样本 F1 从 0.364 提升到 0.516，但精确率 0.727 未达到预设 0.80 门槛；正式验收仍记为未通过。参考标签来自模型复核，不能当成人工金标准确率。整体情绪预测和原因统计是两个独立板块。

## 快速运行

已验证环境：Windows、Python 3.12、NVIDIA RTX 4060 Laptop 8GB，PyTorch 2.7.1 CUDA 11.8。当前推理使用 CUDA 和 BF16，暂未实现完整 CPU 路径。首次安装需下载约 12GB 模型文件，请保留至少 20GB 可用空间。

```powershell
git clone https://github.com/martymi98566-sys/text-sentiment-analysis.git
cd text-sentiment-analysis
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/install_model_assets.py
.\start_review_web.cmd
```

浏览器打开 `http://127.0.0.1:8765/`。网页支持片名、单条英文评论、原文高亮、统计图、HTML 和 JSON 导出。片名由用户填写，不会自动匹配电影数据库。单条分析可能需要数分钟，任务依次执行。

模型包由安装脚本从本仓库 Release 下载；公共预训练模型从 Hugging Face 按固定提交下载，随后进行 SHA256 校验。可到 [Releases](https://github.com/martymi98566-sys/text-sentiment-analysis/releases) 手动下载训练权重 ZIP，再运行：

```powershell
.\.venv\Scripts\python.exe scripts/install_model_assets.py --trained-archive "下载的模型包路径.zip"
.\.venv\Scripts\python.exe scripts/install_model_assets.py --verify-only
```

命令行分析：

```powershell
.\.venv\Scripts\python.exe scripts/analyze_single_review_emotions.py --movie "Example" --text "The acting is excellent, but the plot is painfully predictable."
```

## 阅读入口

- [详细技术说明](docs/TECHNICAL_DESCRIPTION_ZH.md)：模型、数据、弱监督、推理、算法、接口、复现与限制。
- [版本迭代与验证报告](docs/RELEASE_REPORT_20261120.md)：实际指标、正式验收状态、发布验证。
- [原始项目说明](docs/UPSTREAM_README.md)：接手前的模型实验和报告入口。

## 目录

| 路径 | 用途 |
| --- | --- |
| `src/sentiment_analysis/` | 原有分类模型与当前原因分析核心代码 |
| `scripts/` | 数据、实验与应用运行工具 |
| `configs/` | 项目配置、固定策略、模型清单与校验清单 |
| `web/review/` | 本地网页资源 |
| `checkpoints/` | 原项目版本化的序列模型参数 |
| `data/` | 下载模型、必要运行元数据及本地输出；原始评论数据不随本次新增提交发布 |
| `reports/`、`docs/` | 原项目报告、保留的已完成对照结果、版本技术文档 |
| `notebooks/` | 原项目探索笔记；本轮废弃探索未加入 |

本次新增发布不包含聊天记录、凭据配置、失败试训驱动、逐批交接材料、原始影评库或浏览器运行日志。若某个版本号模块被当前推理直接依赖，则仍作为必要运行代码保留。

公开仓库用于下载源码和模型附件。网页程序仍在本机运行，GitHub Pages 不能直接执行本项目的 Python/GPU 推理服务。预训练模型、训练数据与上游代码的使用应遵守各自许可；公开源码不代表已获得所有数据和模型的商业授权。
