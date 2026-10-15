# 真实 Hugging Face 模型扩展路线

Toy model 仍是 00-08 的唯一前提：它把因果掩码、连续 KV cache、调度和资源生命周期缩到可读范围。真实模型是额外的 09+ 路线，用来证明这些概念能迁移到生产格式的权重和 tokenizer；它不能替代前面的基线。

## 选型：Qwen/Qwen2.5-0.5B-Instruct

选择 [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) 作为默认真实模型：它是 Apache-2.0 许可的 0.49B 参数指令模型，具有 24 层、14 个 query heads、2 个 KV heads 的 GQA 架构，并支持中文等多语言。其 GQA、RoPE、RMSNorm 和 SwiGLU 都是从 toy decoder 走向真实推理引擎时值得观察的差异。[模型卡](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)；[Transformers Qwen2 文档](https://huggingface.co/docs/transformers/model_doc/qwen2)。

作为替代，`TinyLlama/TinyLlama-1.1B-Chat-v1.0` 也适合学习 Llama 系结构，但其 safetensors 权重文件约 2.2 GB、定位英语对话；在这台 8 GiB、以中文学习为主的机器上不如 Qwen 0.5B 合适。[TinyLlama 模型卡](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

0.49B 参数以 BF16 保存时仅权重的理论下限约为 `0.49e9 × 2 bytes ≈ 0.98 GB`；仍须为临时激活、CUDA runtime 和 KV cache 留出空间。因此第一轮真实实验限制 batch=1、`max_new_tokens<=64`，不启用量化或自定义 attention kernel，先确保语义可验证。

先安装可选依赖；00-08 不需要它们，也不会因为 import 而下载模型：

```powershell
pip install -e ".[hf]"
```

本章以下“新增”均遵循增量约定：HF 代码只依赖 PyTorch/Transformers，不反向改造 toy model、`ReferenceModelRunner`、`Engine` 或已有 HTTP 服务。

## 09. Hugging Face 基线：先相信原厂实现

**为什么。** 自定义执行器的第一个比较对象不应是肉眼看起来合理的文本，而是 Hugging Face `generate()` 的 token 序列。它同时覆盖 tokenizer、chat template、EOS 与模型前向。

**新增，不改旧代码。** `mini_vllm/hf_backend.py` 是孤立的可选模块；00-08 从不导入它。`lessons/09_hf_baseline.py` 是第一次会下载权重的脚本。

```powershell
pip install -e ".[hf]"
python lessons/09_hf_baseline.py
```

执行前确认磁盘可容纳模型缓存。模型卡建议使用 `AutoTokenizer`、`AutoModelForCausalLM` 以及 `apply_chat_template`；本章严格沿用这一条原生路径作为 oracle。[Qwen 模型卡的示例](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

**验收。** 固定 prompt、`do_sample=False` 和固定 `max_new_tokens` 后，保存 input IDs 与输出 IDs；文本只用于展示，token IDs 才是之后章节的比较基准。本项目已在 RTX 4060 Laptop GPU 上完成 Qwen tokenizer、权重加载、`cuda:0` 放置与一次中文生成的实际检查。

## 10. 检查真实模型的 prefill / decode 等价性

**为什么。** Qwen 的 cache 不再是 toy model 的 `[batch, token, hidden]`：它跨 24 层，K/V heads 数也小于 Q heads。不能根据 toy 模型的张量布局猜测真实 cache。

**背景。** 一个 decode token 的 attention 只有一个 query；整段重算有多个 query。因此两条路径即使语义一致，也会触发形状不同的 GPU 矩阵乘法。BF16 的累加顺序不同会放大为逐元素 logit 的可见差异：本机 Qwen BF16 实测最大绝对差约 `0.164`，但这不代表 cache 索引错误。把很松的 BF16 阈值称作“严格验证”反而会误导学习。

**怎么做。** `HuggingFaceReferenceRunner` 显式调用 `model(..., use_cache=True)`：一次喂完整 prompt 获得 `past_key_values`，再只喂一个 token；然后与 `prompt + token` 的完整前向比较。`lessons/10_hf_cache_parity.py` 专门以 FP32 加载这个 0.5B 模型，令 `rtol=atol=1e-3` 成为有意义的严格 oracle；默认 BF16 服务路径不变。

```powershell
python lessons/10_hf_cache_parity.py
```

先看 `HuggingFaceReferenceRunner.prefill()`：它生成全 1 attention mask、把 token IDs 放入模型 device，并让 Transformers 建立 `DynamicCache`（旧版本可能返回 legacy tuple）。再看 `decode()`：输入严格为 `[batch, 1]`，mask 从 `[batch, L]` 扩为 `[batch, L+1]`，同一个 `past_key_values` 传回模型。位置 ID 由 Qwen 的原生实现根据 cache 已有长度计算；本章不手工复制 RoPE。

**验收。** 输出 `PASS: cached and full logits match`；首层 K/V 均在 `cuda:0`。若想观察 BF16 而非验证它，改回默认 `load()`，同时比较 greedy token ID 与第 11 章的 `generate()` oracle，不能沿用 FP32 的逐元素阈值。

## 11. 真实模型 adapter，而非重写 Engine

**为什么。** 第 03 章已经证明：优化实现必须保留 reference path。真实模型也应遵守相同规则。

**怎么做。** 新增 `HuggingFaceReferenceRunner`，实现与 `ReferenceModelRunner` 等价的 `prefill/decode` 语义，但先直接使用 Transformers 的 `past_key_values`。`HuggingFaceStep` 保存四项最小状态：GPU logits、可变 native cache、GPU attention mask 和唯一回到 CPU 的 `token_id`。不修改 `Request`、`Scheduler`、`Engine`，也不把 Qwen 内部层复制进项目。

```powershell
python lessons/11_hf_runner.py
```

**验收。** `HuggingFaceTextService.complete()` 的 token IDs 与原生 `model.generate(..., do_sample=False)` 一致。下载模型后运行：

```powershell
$env:MINI_VLLM_RUN_HF=1
python -m unittest tests.test_hf_integration -v
```

普通测试中的 `tests/test_hf_runner.py` 不需要 checkpoint，专门验证 adapter 每次 decode 只输入一个 token、mask 和 cache 长度都恰好增长 1。这里尚未把 HF runner 接进第 05 章 scheduler；那必须另起一个“异构 batch + native cache 所有权”的章节，不能暗中重写既有 Engine。

## 12. 从真实 GQA cache 到 mini-vLLM 的 block 设计

**为什么。** Qwen 的 14 Q heads / 2 KV heads 正好说明：KV cache 的显存由 KV heads 而非 Q heads 决定。此时再把第 06a 的逻辑 block table 扩展到真实张量布局才有依据。

**怎么做。** `describe_gqa_cache()` 只根据 config 估算每新增 token 的 K+V 字节数；`summarize_past_key_values()` 同时兼容旧 tuple cache 和新版 `DynamicCache`，只读取 shape/dtype/device。它们不移动、更不重写真实 cache。

```powershell
python lessons/12_hf_gqa_layout.py
```

对默认 Qwen、batch=1、BF16，公式为 `2 × 24 layers × 2 KV heads × 64 head_dim × 2 bytes = 12,288 bytes/token`；14 个 query heads 不进入该式。输出的第 0 层应显示 K/V 形如 `[1, 2, prompt_tokens, 64]`，使公式与真实张量相互印证。

**验收。** 单元测试验证公式确实使用 KV heads，实际 lesson 验证 config 与 cache shape 一致。真正把逻辑 block table 接进 Qwen attention、比较 token/logits、同步释放所有层 block 是后续 **12b** 的新实现；当前课程不会冒充已有 PagedAttention。

## 13a. 真实模型 HTTP 基线：先固定接口，再整合调度

**为什么。** 把 Qwen 直接塞入已有 toy Engine 会同时改变 tokenization、cache 所有权、batch 语义和 HTTP 协议，初学者无法判断问题来自哪里。先提供一个独立、单请求、greedy 的真实模型 endpoint，验证“chat messages → token IDs → UTF-8 JSON”这条边界；它是服务接口 oracle，不是 vLLM 等价物。

**怎么做。** `HuggingFaceTextService` 复用第 11 章 runner，使用 Qwen chat template 和 `/v1/chat/completions`；`create_hf_http_server()` 只绑定 loopback `127.0.0.1:8001`。它没有 streaming、取消、连续 batching 或 block pool，也不替代第 07a 的 toy HTTP 服务。

```powershell
python lessons/13_hf_service.py
# 新开一个 PowerShell：
Invoke-RestMethod -Method Post http://127.0.0.1:8001/v1/chat/completions `
  -ContentType application/json `
  -Body '{"messages":[{"role":"user","content":"用一句话解释 KV cache"}],"max_new_tokens":16}'
```

**验收。** `/health` 不执行 forward；一次 completion 返回 `model`、`token_ids`、`text`、`finish_reason`；第 11 章 integration test 将它的 `token_ids` 与 native `generate` 对齐。后续 **13b** 才定义如何把 HF runner 纳入 Scheduler，并测量 native / adapter / paged cache 的 TTFT、token/s、峰值显存，且每个数字都必须带 batch、上下文、dtype、模型 revision 与 GPU 信息。

这条路线刻意不让 Hugging Face 模型替换 toy model：前者证明真实权重兼容性，后者解释机制；两者一起才构成可信的 mini-vLLM 教程。
