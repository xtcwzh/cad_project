# v5 批量压缩与还原

这是可在 Linux/Windows 运行的 Python 命令行工具，不依赖模型/GPU，不使用二进制编码，不需要原始零件目录。输入为完整的 v4 数据目录，输出独立 v5 目录；不修改 v4。

## 构建

```bash
cd ~/cad_project
python tools/sft_tools/build_v5.py --source data/sft_v4 --out data/sft_v5
```

同一输入和代码会生成同一结果。report.json记录源样本和schema哈希；每件验证序列化后的输入/目标精确展开一致和还原树XML语义等价。任何验证错误都会终止构建；未成功生成本次完整report前，不能使用部分输出训练。重复运行会覆盖指定输出目录中的同名派生文件，因此应使用专用输出目录，不要混入人工文件。变更源数据成员时使用新的空输出目录。

## 格式

输入是`{pool,body}`；输出是`{label_version:5,pool,body}`，其中body展开后是v4标签。`{"$ref":0}`引用当前文档pool的第0项。pool内是可读原始JSON，不能嵌套引用；引用不跨输入/输出文档、不跨零件。空pool和完全展开body也合法，模型不必复刻编码器的最优字典布局。

只合并序列化表示完全相同的值；不舍入浮点数、不混淆整数/浮点数或正负零、不删除任何几何和引用。候选由本件自身重复信息决定，不建立基于验证/测试答案的共享字典。编码器按字符长度选择，实际token收益必须另测。

## 服务器测量

```bash
python tools/sft_tools/check_lengths.py --sft-root data/sft_v5 --tokenizer "$HOME/models/Qwen3.8-27B-bnb-4bit" --context-length 8192 --max-tokens 4096
python tools/sft_tools/profile_tokens.py --sft-root data/sft_v5 --tokenizer "$HOME/models/Qwen3.8-27B-bnb-4bit"
```

check_lengths超预算退出码1是检查结果，不是构建失败。profile的feature字段统计是展开后的语义占用；实际压缩后长度见target_tokens/user_tokens和transport字段。对比v4/v5报告必须保持同一tokenizer、模板和参数。

## 还原

```bash
python tools/sft_tools/restore.py data/sft_v5/simplified/gb10.label.json gb10.restored.json --context data/sft_v5/context/gb10.context.json --schemas data/sft_v5/xml_schemas.json
```

只需该标签、调用方context和冻结schema即可还原，不读取原始答案或其他零件。

## 当前验证与限制

46/46通过精确展开和语义还原；v3/v4/v5共16项单元测试通过。目标总字符数相对v4下降28.11%，输入连同system下降3.85%。最长目标778802→613643字符。

这不等于Creo实际重建通过，也不保证模型能学会字典引用；必须验证引用合法率、语义内容和最终重建。v5仍不是已批准的正式训练数据；当前 QLoRA 读的是 v6，同样未达到可完整微调的长度。现有llm_call/evaluate尚未完整适配v5；恢复入口已支持。数据规模扩大只需批量运行相同编码器，无逐件定制。更复杂零件可能仍需要分步生成。
