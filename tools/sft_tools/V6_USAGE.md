# v6 可逆结构压缩候选

运行 `python tools/sft_tools/build_v6.py`，从 `data/sft_v4` 独立构建 `data/sft_v6`。支持 `--source` / `--out`；不覆盖v4/v5，不需要原始零件或模型。版本化代码可在Linux/Windows批量执行。

基于v5独立文档pool，新增整数等差序列、默认值加例外位置的稀疏数组、同构表按列存储、异构表分组存储并保留原行序。浮点数不参与推导运算。候选按字符长度比较，保留更短文档；因此token数可能不按同比例下降。输出body展开后仍为v4。

构建器对每件的序列化输入和目标做精确展开比较，再做完整树语义还原比较。错误立即终止，不得使用部分输出。报告记录源样本和schema SHA256。重新运行覆盖专用输出目录中的同名文件；源样本成员改变时请使用新空目录。全部46件验证通过，36项单元测试通过。不能据此宣称Creo实际重建或模型生成质量通过。

`restore.py --context ... --schemas ...`支持v6。`profile_tokens.py`支持v6。llm_call/evaluate仍未完整适配v6，不是正式训练就绪版本。

服务器：

```bash
cd ~/cad_project
python tools/sft_tools/build_v6.py
python tools/sft_tools/check_lengths.py --sft-root data/sft_v6 --tokenizer "$HOME/models/Qwen3.8-27B-bnb-4bit" --context-length 8192 --max-tokens 4096
```

第二个命令超限退出码1是正常报告。现已直接打印各split的最短/中位/最长、完整输入/目标预算同时满足的fits数量和累计长度，无需另复制统计代码。minimum_sft_tokens仍为输入+单独目标+8的估算。

本轮v6相对v4目标字符减少30.24%，不是相对v5减少30%。结构压缩收益有限。完整聊天模板下最长样本约76.7万token（`外人字形齿轮gear`），train最短约1.1万、最长约6.1万token，全部超过已通过的三卡8K QLoRA；三卡与五卡按层分片在32K/64K合成序列上均OOM。不能把v6当作已可完整微调的格式。更大幅缩减需要对程序可生成的几何信息建立可验证的恢复规则，或设计8K能装下的分步任务，不允许直接截断/删除引用。
