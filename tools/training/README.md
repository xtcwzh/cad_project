# CAD QLoRA：服务器检查、显存阶梯测试和训练

## 当前：四卡 PCIe 显式分层实验

### 运行诊断（保持四卡 / 32K / fp32 / chunked1024）

`configs/qlora_probe_4gpu_chunked_1024_diag.json` 与 chunked1024 相同，只打开运行诊断。
记录 SDPA 调用处 Q/K/V 与处理后 attn_mask 的 shape/dtype/device，以及 `is_causal`、
`enable_gqa` 是否被传入；记录 decoder 层号、`block_type`、以及
`no_grad` / `grad` / `checkpoint_recompute`（由 `grad_enabled` 与 checkpoint 栈帧测得）；
用 aten `TorchDispatchMode` 记录名字含 `scaled_dot_product` 的前向/反向算子。
OOM 时保存有界 CUDA 分配历史（默认最多 2048 条）和失败分配的 Python 栈，不含激活。
SDPA 包装使用 `torch.compiler.disable`，dispatch mode 会 graph break；**此跑的显存峰值不能与未诊断探测比较**。
未出现在日志里的字段不要当成实测。

```bash
python -B -m unittest discover -s tools/training -p 'test_runtime_diagnostics.py' -v
python tools/training/probe_training_memory.py --config configs/qlora_probe_4gpu_chunked_1024_diag.json --lengths 32768
```

输出在 `outputs/qlora-4gpu-chunked-1024-diag/`：`diagnostics.json`、`events.jsonl`，
摘要写入 `result.json` 的 `diagnostics`。测试只用小型 CPU 张量。

### MLP 按 token 分块实验（32K 上下文不变）

本轮 2048-token 分块已消除原先 1.06 GiB 的单个 MLP gate 中间张量，但四卡在第一步
反向阶段整体接近满载（各卡约20.8–22.5 GiB），GPU0在申请768 MiB时只剩0.33 GiB。
这属于新的全局峰值，不是原始 gate 张量错误。下一档配置为
`configs/qlora_probe_4gpu_chunked_1024.json`，只把内部块从2048降到1024，其他设置
完全不变；预计速度更慢，先验证能否越过第一步。

上传 `chunked_mlp.py`、`training_core.py`、`test_chunked_mlp.py` 到
`tools/training/`，以及 `configs/qlora_probe_4gpu_chunked.json`。
保留此前修复过的 `device_plan.py`。新配置回到 `[12,20,20,12]` 以使用全部四卡
承载decoder层；增加 `mlp_chunk_tokens=2048`。原配置保留不变，输出独立保存。

MLP沿序列维度分块并按顺序拼接，不切样本、不切注意力、不改变标签或上下文。
开启梯度时，每块使用原生PyTorch非重入checkpoint，避免宽中间激活在拼接之前
累计保留。外层Unsloth整层检查点仍开启；保留Accelerate的外层设备钩子。
仅分块循环使用 `torch.compiler.disable`，这是兼容性优先的实验，速度可能下降。
没有可识别的原生checkpoint备份时明确报错，不静默调用Unsloth重入替代函数。

先在服务器环境执行CPU数值测试（不加载27B模型），成功后再探测：

```bash
python -B -m unittest discover -s tools/training -p 'test_chunked_mlp.py' -v
python tools/training/probe_training_memory.py --config configs/qlora_probe_4gpu_chunked.json --lengths 32768
```

测试覆盖完整与分块输出、输入梯度、全部LoRA矩阵梯度、非整除尾块、外层重入
checkpoint嵌套、冻结输入、设备包装器保留及不累计保存宽中间激活。测试使用小型
CPU MLP；不代表服务器Unsloth/4bit内核兼容性或32K显存验证已完成。
日志应出现 `Chunked MLP` 和 `chunked_mlp_layers=64` 对应的JSON字段。
成功仍以完成3步、有限loss/grad_norm、LoRA更新及adapter回读为准。
32K合成通过后用相同配置 `--mode real --lengths 32768` 检查最长可容纳的真实训练
样本，再评估64K；不将32K子集探测当作全量数据训练已就绪。

### 32K 反向 OOM 后的独立输出头实验

`configs/qlora_probe_4gpu_head_only.json` 使用 `[16,24,24,0]`：GPU0 放
embedding/视觉塔及0–15层，GPU1 放16–39层，GPU2 放40–63层，GPU3 仅放
最终norm和lm_head（无decoder层）。保留原 `[12,20,20,12]` 配置以便对照。
上传更新后的 `device_plan.py` 和新配置后运行：

```bash
python tools/training/probe_training_memory.py --config configs/qlora_probe_4gpu_head_only.json --lengths 32768
```

此实验将末卡decoder/FFN的反向重计算移到前三卡，未修改模型结构、精度、rank、长度
或梯度检查点。原末卡在反向重计算MLP时OOM；其他卡尚未完成反向，原峰值不能当作
新分配的容量证明。输出头自身显存、跨卡梯度和完整三步更新仍需服务器验证。
报错临时张量为FP16也说明FP32配置不代表每个内核的中间张量都是FP32。

### 原始四卡配置与共用说明

2026-09-18 修正：显式映射不再包含根键 `"": 0`，单独指定 `rotary_emb`。
根映射可能使 Accelerate 先递归安装 GPU0 执行钩子，造成后续层的子模块输入
回到 GPU0、权重却在 GPU1 的错误。现在除参数位置外也检查子模块执行钩子
（含 SequentialHook）的设备。保留原精度和编译设置，重新启动32K探测进程；
无需删除编译缓存。该修正仍需在服务器验证前向、反向和显存峰值。

上传 `tools/training/training_core.py`、`tools/training/device_plan.py` 和
`configs/qlora_probe_4gpu.json`，替换服务器对应文件前保留服务器现有版本备份。
本次已保留服务器验证过的 FP32 加载、禁用 Trainer FP16 autocast 的做法。
四卡配置固定 `precision=fp32`，NF4 主体权重量化保持不变；embedding/head 的实际
存储 dtype 和量化模块 compute_dtype 由结果报告记录，不假设所有运算精度相同。

机器没有 NVLink，只有 PCIe。使用一个 Python 进程按连续层分片，不使用 torchrun。
默认物理 GPU 0/1/2/3：GPU0 放视觉塔、embedding、语言层0–11；GPU1 放12–31；
GPU2 放32–51；GPU3 放52–63、最终norm和lm_head。第五张卡不使用。
`layer_counts=[12,20,20,12]` 是为首尾卡预留容量的实验起点，不是实测最优分配。
模型必须是64层、未绑定embedding/head的qwen3_5；加载、LoRA创建和Trainer初始化后
均检查实际参数设备。模块名称不匹配会停止，不能把此类错误解释为显存不足。

```bash
python tools/training/inspect_environment.py --config configs/qlora_probe_4gpu.json
python tools/training/probe_training_memory.py --config configs/qlora_probe_4gpu.json --lengths 32768
```

32K成功后再分别执行64K合成和真实样本测试：

```bash
python tools/training/probe_training_memory.py --config configs/qlora_probe_4gpu.json --lengths 65536
python tools/training/probe_training_memory.py --config configs/qlora_probe_4gpu.json --mode real --lengths 65536
```

每次结果在 `outputs/qlora-4gpu/probe-模式-时间/长度/result.json`。
检查实际设备分布、`training_memory.per_gpu`、有限loss/grad_norm、LoRA更新和adapter回读。
比较速度应参考编译结束后的step耗时；更多卡不一定必然更慢，也不会合并为统一显存。
分层不能拆分单层的长序列激活，四卡32K/64K容量仍需服务器实测。
合成测试成功不等于完整数据集可训练，真实测试只取长度内最长的一个train样本；
正式训练仍需全部train/val的完整模板长度预检，禁止静默截断或跳过超长数据。

以下三卡说明为历史配置，不要与本次四卡配置混用。

## 新增：三卡64K分片实验（优先于下文单卡说明）

上传更新后的`tools/training/`与`configs/qlora_probe_3gpu.json`。此独立配置选择物理GPU 0、1、2；进程内编号对应cuda:0、1、2。使用单进程`device_map="balanced"`，不是DDP，不使用torchrun。每卡加载分配预算10GiB，不限制训练中间张量的总显存。关闭embedding CPU卸载以先验证纯GPU按层分片；单卡配置仍保留原卸载设置。

```bash
python tools/training/inspect_environment.py --config configs/qlora_probe_3gpu.json
python tools/training/probe_training_memory.py --config configs/qlora_probe_3gpu.json --lengths 65536
```

测试3次optimizer更新，输出独立目录`outputs/qlora-3gpu/probe-synthetic-时间/65536/`。`result.json`含加载后/LoRA后/Trainer后的实际hf_device_map、每卡参数存储与显存；training_memory.per_gpu记录训练阶段各卡峰值。时间同步所有选中GPU。加载后必须使用全部选中GPU，禁止意外CPU/meta参数放置；Trainer必须识别model_parallel并禁用DataParallel复制。任何断言失败表示配置/版本需适配，不是GPU容量不足。64K仍可能因单层计算或跨卡内核不兼容失败，尚无服务器验证。

如物理0/1/2被占用，可改gpu为另一组三个不重复编号。当前仅授权三卡测试，不默认使用剩余两张。单次超时7200秒；失败或超时停止，不自行降长度。不要启动多个并行probe占同一组卡。

合成64K成功后，完整真实train样本复验：

```bash
python tools/training/probe_training_memory.py --config configs/qlora_probe_3gpu.json --mode real --lengths 65536
```

采用同一模型加载逻辑；无需重新生成数据。probe adapter只作测试。本地7项测试通过，不代表三卡GPU训练成功。

上传 `tools/training/` 和 `configs/qlora_probe.json` 到服务器项目对应目录。已有模型和`data/sft_v6`无需重新传输。所有命令默认当前位于项目根目录。路径参数统一支持`~`；配置中的相对路径相对于项目根目录。

## 默认配方

- 本地`~/models/Qwen3.8-27B-bnb-4bit`，不联网下载、不安装或升级依赖。
- 单卡GPU 0；如被占用，在配置中修改`gpu`。不要用torchrun启动；当前没有多卡分片功能。
- 使用Unsloth FastModel，4bit冻结基座，语言注意力及MLP LoRA，视觉层冻结；r=16、alpha=16。
- TITAN RTX采用FP16；只有sm>=80且PyTorch支持时才采用BF16。不强制安装FlashAttention。
- `offload_embedding=true`，沿用官方可选内存卸载方案；实际平台支持与占用需实测。如不支持会明确失败/由框架处理，不声称已经生效。
- 使用Unsloth梯度检查点，micro-batch=1、adamw_8bit。探测梯度累积1，正式训练配置默认4。
- thinking=false。对完整system/user/assistant聊天模板编码，严格核对prompt字符串和token前缀。只监督assistant后缀（含模板收尾）；不猜分隔符位置、不静默截断。模板前缀不同会报错，需要根据实际模板修复，不得绕过。

## 1. 环境与精确长度

```bash
python tools/training/inspect_environment.py
python tools/training/prepare_training_data.py --length 65536
```

输出`outputs/qlora/environment.json`和`exact_lengths.json`。环境检查只读取模型配置，不加载权重；精确长度检查只加载本地tokenizer。检查全部三个split，逐件记录精确总长度、监督token数和超限情况。这里不使用旧的“prompt+target+8”估算。

## 2. 合成序列容量测试

2026-09-17服务器实测8K在第一次前向的FLA GatedDeltaNet `v_new=torch.empty_like(u)` 分配96MiB时OOM，未完成optimizer step，不能声称8K已通过或直接推断2K也不可行。新增base_loaded/lora_created/trainer_initialized显存快照与参数设备统计，失败保留phase和failure_memory。此时先用`--lengths 2048`诊断整个前后向链路；2K为容量实验，不是把真实CAD标签截成2K。成功后再逐级4K/8K；超长目标仍未解决。

先只运行8K确认加载、内核和优化器路径，再扩大：

```bash
python tools/training/probe_training_memory.py --lengths 8192
python tools/training/probe_training_memory.py --lengths 16384 32768 65536
```

这不是CAD训练：人工构造指定长度的token，后半段参与loss，不保存为正式数据。每个长度启动独立进程，完成3次optimizer更新；验证有限loss/梯度日志、有限LoRA值、采样LoRA参数变化、保存adapter后用PEFT重新读取并逐tensor与内存比较。adapter读回不等于重新加载整套基座推理验证。

报告包含加载时间、每个优化器步耗时、PyTorch峰值allocated/reserved显存、参数更新和读回结果。编译可能在首步发生；allocated/reserved不涵盖所有驱动/NCCL等外部内存，也不是nvidia-smi整卡峰值。当前不自动把某一级“passed”宣布为长时间稳定训练。

每次输出新目录`outputs/qlora/probe-synthetic-时间/`，内含每长度的worker.log、result.json、adapter和汇总summary.json。超时默认3600秒，首次编译慢可以改配置；timeout不是OOM。OOM/通用错误/超时均单独记录，停止继续探测更长序列。每个成功长度会保留adapter，需关注磁盘空间；不导出56GB合并模型。

## 3. 完整真实样本复验

例如32K合成测试成功后：

```bash
python tools/training/probe_training_memory.py --mode real --lengths 32768
```

自动选训练集中不超过该上限的最长完整样本，报告零件名与实际token数，重复该样本完成3步。样本可能短于上限；该结果不代表填满上限时一定成功。没有合适样本明确返回no_fitting_real_sample，不截断答案，不使用val/test训练。probe adapter仅供测试，不能作为正式CAD成果。

## 4. 正式训练

只有对应长度通过合成容量测试和真实样本复验后才执行。以下`65536`是命令示例，不是已验证推荐值，当前验证集还可能超出此长度：

```bash
python tools/training/train_qlora.py --length 65536 --run-name pilot-v6
```

入口重新编码并检查所有train/val，任何超长样本均拒绝启动，不静默过滤。test只做长度检查，不参与训练或验证loss。服务器实测前不要为了通过检查而随意把length设到很大。新run-name不覆盖已有目录；preflight失败也可能留下诊断目录，再试用新名称。

训练结束保存标准PEFT adapter和tokenizer、训练/验证loss、Trainer状态、实际配置和冻结schema。过程每20步保存检查点、最多保留2份。验证loss不等于JSON、特征树或Creo重建正确率；生成评测仍需后续适配v6。

恢复时使用同一run-name和长度、同一模型路径/配置/数据，显式给出该run目录内的checkpoint：

```bash
python tools/training/train_qlora.py --length 65536 --run-name pilot-v6 --resume outputs/qlora/train-pilot-v6/checkpoint-20
```

checkpoint必须实际存在。全量数据阶段沿用同一模型基座和脚本；不能把测试集混入训练。若单卡无法满足长度，再另行实现和测试分片训练，不把5卡当作自动共享120GB。

## 验证范围与参考

本地完成6项CPU单元测试：完整模板与loss mask、前缀/token边界失败拒绝、超长数据保留、跨split重复拒绝、单卡参数限制、报告写入。未在本地执行27B加载/CUDA训练；安装包存在也不证明sm_75内核支持，实际失败日志是兼容性诊断依据。

官方接口依据：[Unsloth Qwen3.8训练文档](https://unsloth.ai/docs/models/qwen3.8/train)。这里使用Transformers Trainer消费已经tokenize并掩码的样本，避免SFTTrainer再次套模板或截断。未来框架升级可能改变接口，遇到问题保留environment.json和worker.log。
