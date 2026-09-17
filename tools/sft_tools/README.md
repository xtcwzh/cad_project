# 数据链路 v3 / v2（2026-09-17）

当前 QLoRA 读 `data/sft_v6`，但还不能按完整零件开训。2026-09-17：三卡合成 8K 通过；三卡/五卡合成 32K 与 64K 均在注意力/反传 OOM。v6 train 最短约 1.1 万 token，超过已验证的 8K。硬件结论见 `tools/training/README.md`。v6 构建见 [V6_USAGE.md](V6_USAGE.md)；v5 见 [V5_USAGE.md](V5_USAGE.md)。下文仍记录 v3/v2：`llm_call.py`、`evaluate.py`、`check_lengths.py` 默认根目录仍是 `data/sft_v3`，查更新版本必须显式传 `--sft-root`。

## 不可变设计前提

这条管线面向 46 件试点、874 件零件库以及未来上万件工厂数据。压缩、编码、还原和校验必须由版本化代码批量运行并可复现，不接受逐件手改标签、人工补 XML 或依赖原始真值的隐式后处理。压缩表示必须包含恢复 Creo 所需的信息；缺省值只能按明确规则确定性补回。每次变换都要有原始导出对照、失败清单和可追溯版本。转换测试通过不等于 Creo 重建成功，后者必须单独闭环验证。

## v3 当前默认工作格式

`build_v3.py` 从原始树生成 `data/sft_v3`，v2 保留作对照。`llm_call.py`、`evaluate.py`、`check_lengths.py` 默认读取 v3；查看 v2 时显式传 `--sft-root data/sft_v2`。

- 简单拉伸、旋转、孔、倒角、阵列、基准的 XML 改为 `op + schema + params`。孔即使内部是拉伸/旋转形式，语义操作仍标为 hole。无 XML 的圆角标记 source_xml_missing，不伪造参数。
- XML 库只从29个训练零件提取精确节点/属性结构，不存任何叶子文本参数；不读取验证/测试数据创建结构。参数路径可读、值留在标签内。仅按明确的格式规则补空值、整数0、double的0.00，保留非默认值的词法精度。
- 未见结构、扫描等暂不支持的复杂 XML 使用 raw_xml 透传，不套用别的零件模板。同一结构目录/槽位说明向所有划分公开，不按测试答案挑选提示。
- 重复对象数组改为 `$table`/`$records`，保留缺失字段与null的区别、顺序、数值精度。几何映射暂不删除，旧插件需要它们；尚未实现语义面/边选择器或阵列成员去展开。
- 名称、单位、密度、材料移到显式 provided_context。该上下文是调用方给定的元数据，试点从原导出注释中取得，不是从几何推断，也不包含 XML/草图/特征树。部署时需调用方提供；评测不会暗中用整棵真值补预测。
- XML序列化改为规范形式：允许声明/缩进/空元素写法变化；比较节点、属性、叶子文本和顺序。其他JSON字段保持原值。尚未通过Creo实际导入验证。
- `payload_guard.py` 在生成/还原阶段拒绝快照、base64编码字段、data URI及长base64类载荷。原始导出快照不删；训练产物不含这些载荷。

v3验证命令（项目根目录，前缀仍为本机cad环境python）：

```text
tools/sft_tools/build_v3.py
tools/sft_tools/test_semantic_v3.py
tools/sft_tools/verify_payloads.py
tools/sft_tools/check_lengths.py
tools/sft_tools/llm_call.py --split val --tag v3_mock --mock
tools/sft_tools/evaluate.py --split val --tag v3_mock
tools/sft_tools/restore.py data/sft_v3/simplified/gb10.label.json data/sft_v3/gb10.checked.json --context data/sft_v3/context/gb10.context.json --schemas data/sft_v3/xml_schemas.json
```

重新生成v3会更新派生文件，既有预测应重新生成或另取tag。XML结构库及codec必须与数据和模型检查点一起冻结保存。

**当前v3约缩小53%，仍不是全部样本已满足训练长度的最终格式。最长复杂零件输出仍约80万字符；固定schema说明也占输入。必须在实际模型环境运行token检查。** 后续大幅缩短需重建端支持语义几何引用和阵列展开；不能仅删除这些字段。不要将转换测试通过当成Creo成功。

## v2 保真参考

此版本先解决信息损坏。它是**保真参考格式**，不是已经优化好的紧凑 SFT 标签。
将 v1 的按类型最短 XML 模板替换、尺寸重编、草图/几何省略全部停用。
v1 已不可逆丢失信息，restore 会明确拒绝，必须从原始导出重新生成。

## 格式和边界

- `label_version=2`；`feats` 中 `type` 保留完整类型名称，`type_id` 保留真实整数。
- `dims` 为完整尺寸对象数组，保留 ID、符号、精度、公差；草图实体保留原 ID 和类型。
- XML、阵列、约束、几何映射、引用路径、模型参数及未知扩展字段都保留。
- 只有 `native_snapshot` 被删除，`import_mode` 强制设为 `feature_rebuild`。
- 不依赖 templates.json，也不在预测还原时读取真值来补字段；原始空 XML 保持空。
- 输入包含 evidence 的完整现有字段，去掉重复摘要 `llm_brief`。它仍不是精确 BREP 的无损表达；没有新增 STEP 边界曲线提取。
- 本轮没有修改 Creo 插件，也没有证明上游空 XML、复杂特征恢复问题已解决。

## 本地已生成数据

`D:\cad_project\data\sft_v2` 包含 clean、simplified、samples、restore 和验证报告。
原始 `D:\pilot_dataset` 及旧派生数据保持原样。仍保留原 train29/val5/test12 划分。

46 件直接对照原始 JSON 全字段相等（除上述两项有意变更）。这证明转换保真，**不证明重建成功或模型可学习**。
标签约 8.75 MiB，平均输出 199,134 字符；最长输出 1,508,984 字符。
不能使用旧版 8K token 输出预算或此前按旧标签估算的显存直接训练。`report.json` 明确标注 training_ready=false。

## 验证与命令

在项目根目录，用 `C:\Users\wzh\.conda\envs\cad\python.exe -B` 运行以下脚本：

```text
tools/sft_tools/prepare_data.py
tools/sft_tools/test_roundtrip.py
tools/sft_tools/test_pipeline_v2.py
tools/sft_tools/audit_information_loss.py --out data/sft_v2/information_loss_audit.json
tools/sft_tools/check_lengths.py
tools/sft_tools/llm_call.py --split val --tag mock_v2 --mock
tools/sft_tools/evaluate.py --split val --tag mock_v2
```

已有预测文件默认不会被覆盖；需要复跑时换 tag，或明确传入 `--overwrite`。
mock 只检查程序通路，结果含 mode=mock，不能当模型能力结果。

真实模型调用需先在部署环境使用实际本地 tokenizer：

```text
tools/sft_tools/check_lengths.py --tokenizer <模型目录> --context-length <服务上限> --max-tokens <输出预算>
tools/sft_tools/llm_call.py --split val --tag <模型标识> --tokenizer <模型目录> --context-length <服务上限> --max-tokens <输出预算>
```

部署环境应安装与模型相容的 transformers。tokenizer 仅从本地加载，不自动下载。
调用器只发送 system/user；预算不足拒绝执行，服务返回 length 也不会当正常完成。
token 检查假设部署端使用同一聊天模板与思考设置；SFT 框架自身的最终模板也需核对。

## 后续顺序

1. 在 Creo 中比较原始去快照树与 v2 还原树的逐特征重建结果。
2. 基于本版保真参考，把 XML 转为可验证的结构化操作参数，设计更短的标签及确定性还原器；不能再直接删除字段或使用整棵真值补丁。
3. 对新的紧凑格式验证操作语义、引用一致性、实际 token 长度与 Creo 几何，再开始正式微调。
# v4 压缩候选（2026-09-17）

最新v6结构压缩实验见 [V6_USAGE.md](V6_USAGE.md)。`build_v6.py` 从v4独立生成v6；保留v5。列式/稀疏/整数序列编码不改变数值，收益仍需实际token检查。

后续 v5 精确去重候选已实现：见 [V5_USAGE.md](V5_USAGE.md)。`build_v5.py` 从 v4 批量构建 v5，保留 v4；还原入口已支持 v5，真实 token 和 Creo 验证仍待完成。

`python tools/sft_tools/build_v4.py` 从已有 `data/sft_v3` 构建隔离的 `data/sft_v4`，无需 Windows 原始导出目录。可用 `--source`、`--out` 指定路径。每件均通过序列化后还原对照，否则报错。输出没有覆盖 v3。

v4 使用冻结 XML 槽位编号和紧凑 XML 树，保留所有数值精度及引用；schema 与 v3 相同。`restore.py --context ... --schemas ...` 支持 v4。公共说明和样本已由 builder 同步生成，不要向 samples 再重复添加 system_prompt。

服务器检查：

```bash
python tools/sft_tools/check_lengths.py --sft-root data/sft_v4 --tokenizer "$HOME/models/Qwen3.8-27B-bnb-4bit" --context-length 8192 --max-tokens 4096
python tools/sft_tools/profile_tokens.py --sft-root data/sft_v4 --tokenizer "$HOME/models/Qwen3.8-27B-bnb-4bit"
```

长度超限时第一个命令退出码1属正常报告。`token_profile.json` 的组件分别编码，仅用于定位占用，不能相加当作精确整条长度。此处 `--context-length 8192` 只是测量预算；2026-09-17 三卡 QLoRA 仅通过合成 8K，完整零件仍超限。v4仍是候选，尚未通过真实token预算与Creo重建；llm_call/evaluate对v4的完整支持仍待实现。
