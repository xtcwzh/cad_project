# CAD QLoRA：服务器检查、显存阶梯测试和训练

## 2026-09-17 实测结论

本机 5×TITAN RTX 24GB（sm_75，无 BF16/FA2）。Unsloth 4bit Qwen3.8-27B，单进程按层分片（`device_map=balanced`），不是 DDP，不用 torchrun。Turing 上必须 FP32，Trainer 不得开 FP16，否则 inf `grad_norm`。

| 配置 | 长度 | 结果 | 记录 |
|---|---|---|---|
| 单卡 | 8192 | 加载后 FLA 前向 OOM | `outputs/qlora/probe-synthetic-20260917-073908` |
| 三卡，预算 10GiB | 65536 | 加载失败，4bit 层被派到 CPU | `outputs/qlora-3gpu/probe-synthetic-20260917-080628` |
| 三卡，预算 20GiB | 8192 | 通过：3 step，loss 有限，adapter 回读成功 | `outputs/qlora-3gpu/probe-synthetic-20260917-082945` |
| 三卡，预算 20GiB | 32768 | 训练第一步 OOM（GPU2） | `outputs/qlora-3gpu/probe-synthetic-20260917-084737` |
| 三卡，预算 20GiB | 65536 | 训练第一步 OOM（GPU1 申请 4GiB） | `outputs/qlora-3gpu/probe-synthetic-20260917-084321` |
| 五卡，预算 12GiB | 32768 | 训练反传 OOM（GPU4） | `outputs/qlora-5gpu/probe-synthetic-20260917-085830` |

五卡 32K 已把权重装上。失败时 GPU0 几乎空转（仅视觉塔，峰值约 0.87GiB），GPU4 堆了第 38–63 层（26 层）加 `lm_head`，峰值约 22.6/23.5GiB。OOM 申请 640MiB，等于一条 `32768×5120×FP32` 隐状态。按层分片不切开序列；加卡和改 `weight_budget_per_gpu` 都不能把 32K 激活从拥有该层的那张卡上挪走。

完整 v6 train 精确长度约 1.1 万–6.1 万 token，最短已超过 8K。8K 合成通过只证明训练环可用，不能训完整零件。不要再重复同一套 `balanced` 的 32K/64K 探测。

当前不要启动 `train_qlora.py`。尚未通过任何能装下完整 train 样本的合成+真实复验。

## 还值得做的

1. 手写 `device_map`：把语言层分到 GPU0，把 `lm_head` 移出 GPU4，64 层尽量均分。可能让 32K 挤过；64K 隐状态约 1.28GiB，仍可能爆。
2. 序列并行（Ulysses / context parallel）：沿长度切分，当前脚本未实现。
3. 激活卸到 CPU（主机约 377GiB 内存）：可能极慢，FLA/注意力内核未必支持。
4. 按特征/操作把样本拆到 8K 能装下的步，不删几何。这与再做有损压缩不是一回事。
5. 换 80GB 级 GPU，或换更小模型。

不值得再试：第六张卡、把预算改回 20GiB、再跑一遍五卡 `balanced`。

## 配置与命令

配置：`configs/qlora_probe.json`（单卡）、`configs/qlora_probe_3gpu.json`、`configs/qlora_probe_5gpu.json`。`weight_budget_per_gpu` 是加载规划上限，不是每卡要装满；三卡 10GiB 会 CPU dispatch，五卡 12GiB 加载成功但训练仍 OOM。多卡必须 `offload_embedding=false`。不要用 torchrun。不要并行开两条 probe。

```bash
python tools/training/inspect_environment.py --config configs/qlora_probe_5gpu.json
python tools/training/probe_training_memory.py --config configs/qlora_probe_5gpu.json --lengths 32768
```

日志在 `outputs/qlora-5gpu/probe-synthetic-时间/长度/`。`result.json` 含 hf_device_map、每卡参数存储和 `failure_memory`。加载后必须用满所选 GPU，禁止意外 CPU/meta 放置；Trainer 必须走 model_parallel，禁用 DataParallel。断言失败表示配置/版本问题，不是“卡还没装满”。

## 默认配方

- 本地`~/models/Qwen3.8-27B-bnb-4bit`，不联网下载、不安装或升级依赖。
- 单进程按层分片；未实现序列并行。多卡不能把一条超长样本的激活按序列切开。
- Unsloth FastModel，4bit 冻结基座，语言注意力及 MLP LoRA，视觉层冻结；r=16、alpha=16。
- TITAN RTX 无硬件 BF16；Qwen3.5 路径 Unsloth 用 FP32。Trainer 不得开 FP16 autocast。不强制 FlashAttention。
- 单卡配置可 `offload_embedding=true`；多卡分片必须为 `false`。
- Unsloth 梯度检查点，micro-batch=1、adamw_8bit。探测梯度累积 1，正式训练配置默认 4。
- thinking=false。完整 system/user/assistant 模板编码，只监督 assistant 后缀；不猜分隔符、不静默截断。

路径支持`~`；相对路径相对项目根目录。已有模型和`data/sft_v6`无需重传。

## 1. 环境与精确长度

```bash
python tools/training/inspect_environment.py
python tools/training/prepare_training_data.py --length 65536
```

输出`outputs/qlora/environment.json`和`exact_lengths.json`。环境检查只读模型配置，不加载权重；精确长度只加载本地 tokenizer。检查全部三个 split。不使用旧的“prompt+target+8”估算。已测：train 最短超过 8K，故 `--length 8192` 的正式训练会被拒绝。

## 2. 合成序列容量测试

这不是 CAD 训练：指定长度的人造 token，后半段参与 loss。每个长度独立进程、3 次 optimizer 更新；检查有限 loss/梯度、LoRA 有限且有更新、PEFT adapter 回读。adapter 回读不等于整模推理验证。

```bash
python tools/training/probe_training_memory.py --config configs/qlora_probe.json --lengths 8192
python tools/training/probe_training_memory.py --config configs/qlora_probe_3gpu.json --lengths 8192
```

单卡 8K 已 OOM，不必再从 2K 起跳来“证明链路”；三卡 8K 已通过。32K/64K 在三卡和五卡 `balanced` 下已 OOM，不要当未测过的下一步。超时不是 OOM。每个成功长度会留 adapter，注意磁盘；不导出合并基座。

## 3. 完整真实样本复验

仅当该长度的合成探测通过后：

```bash
python tools/training/probe_training_memory.py --config configs/qlora_probe_3gpu.json --mode real --lengths 8192
```

8K 真实复验预期 `no_fitting_real_sample`（train 最短约 1.1 万）。不截断答案，不用 val/test 训练。probe adapter 不能当正式 CAD 成果。

## 4. 正式训练

不要在 32K/64K 合成失败后强行开训。入口会重新编码并拒绝任何超长 train/val，不静默过滤。test 只做长度检查。

```bash
python tools/training/train_qlora.py --config configs/qlora_probe_3gpu.json --length 8192 --run-name pilot-v6
```

当前 8192 会被长度检查拒绝。更大 `--length` 尚未通过合成探测。验证 loss 不等于 JSON/特征树/Creo 正确率。

## 验证范围与参考

本地 7 项 CPU 单元测试：完整模板与 loss mask、前缀/token 边界拒绝、超长保留且跨 split 重复拒绝、GPU 编号解析、拒绝 torchrun、原子 JSON 写入。CUDA 结论以服务器 worker.log 为准。

官方接口：[Unsloth Qwen3.8 训练文档](https://unsloth.ai/docs/models/qwen3.8/train)。使用 Transformers Trainer 消费已 tokenize 并掩码的样本，避免 SFTTrainer 再套模板或截断。
