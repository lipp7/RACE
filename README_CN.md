# RACE 开源目录说明

完整英文说明见 [README.md](README.md)，基线命令见 [BASELINES.md](BASELINES.md)。

本目录可整体复制到其他机器。数据、缓存和输出均使用项目内相对路径；
不包含原机器的目录、账号信息、历史运行记录、图像数据集和大模型权重。
MIT 许可证已加入，保留的第三方代码包含其原许可证。

建议按以下顺序运行：

1. 在本目录安装 `requirements.txt` 中的依赖。
2. 按英文 README 的结构把数据放在 `data/` 下。
3. 用 `python -m race.extract --dataset cub --baseline-caches` 提取十视图。
4. 用 `python -m race.train --dataset cub --seed 1 --output runs/race_cub_s1` 训练。
5. 用 `python -m race.evaluate --run runs/race_cub_s1` 评估验证集最优模型。

不需要运行旧论文目录中的任何脚本。`maft/` 中保留的内部名称用于兼容
已有模型权重；统一公开入口位于 `race/`。

运行 `python -m unittest discover -s tests -v` 可检查核心公式、反向传播、
概念响应保持及“训练后移动目录再评估”的完整流程。此测试使用合成小数据，
不代替五个数据集上的完整 GPU 实验。
