# mini-vLLM 跟读教程

> 目标：在一张消费级 NVIDIA GPU 上，实现一个**单模型、单卡、可流式输出**的推理服务。它要正确地处理多个请求、增量维护 KV cache、做连续批处理并回收内存。它不试图实现分布式并行、量化 kernel、投机采样或 vLLM 的所有优化。

## 怎样使用这份教程

每章按同一顺序进行：先读“为什么”，再只看该章指定文件并运行命令，然后完成验收。不要跳过测试；后面的正确性建立在前面的不变量上。

| 标记 | 含义 |
| --- | --- |
| ✅ | 已由维护者按本文步骤实现并在本机 CUDA GPU 上验证 |
| ⏳ | 课程设计已固定，待前一章完成后才会实现和验证 |

**通用验收命令**（从仓库根目录运行）：

```powershell
python -m unittest discover -s tests -v
```

每次修改后，至少运行该命令。测试不是“代码能运行”的替代品：它要比较优化前后的数值或状态，证明优化没有改变语义。

课程的依赖边界与“只新增、不倒灌”规则见[增量约定](incremental-contract.md)。它也是后续修改代码时的优先级：宁可增加一个小适配器，也不把未来章节的机制塞回前置模型。

当前整合路径为：`Request (CPU)` → `Scheduler (CPU)` → `PooledModelRunner (H2D + GPU forward + D2H token)` → `KVCachePool (GPU)` → `Engine / Service (CPU)`。第 02 章的 `ReferenceModelRunner` 则保留为不依赖 pool 的学习与正确性基线。章节 06 的 block table 已单独实现和测试；它展示分页映射的正确性，尚未替换主路径使用的连续 cache pool。

## 完成后的整体结构

```text
HTTP client
   │ POST /v1/completions (CPU JSON)
   ▼
InferenceService ─── Request state machine ─── waiting queue
   │                                                │
   │                                           Scheduler (CPU)
   │                                                │ execution plan
   ▼                                                ▼
stream / poll token ◄── Engine ◄────────────── PooledModelRunner
                                                 │ H2D: token IDs + metadata
                                                 ▼
                          GPU: ToyAttentionLM + preallocated KVCachePool
                                                 │ D2H: only sampled token IDs
                                                 ▼
                                         Engine updates request lifecycle
```

阅读代码时始终问两个问题：此数据是否必须跨 CPU/GPU 边界？它的所有权何时结束？前者决定延迟，后者决定 cache 是否泄漏。

## 00. 看见 CPU 与 GPU 的边界 ✅

**要解决的问题。** 推理系统不是“把模型 `.cuda()`”就结束了。请求解析、tokenizer、队列、HTTP 都在 CPU；模型权重、激活值和 KV cache 应留在 GPU。跨越边界会发生实际的数据复制。

**阅读与执行。** 阅读 `mini_vllm/device.py` 和 `lessons/00_cpu_gpu.py`，然后执行：

```powershell
python lessons/00_cpu_gpu.py
```

输出必须依次表明：输入在 `cpu`、矩阵结果在 `cuda:0`、采样 token 最终回到 `cpu`。第一次 CUDA 调用有上下文初始化成本，示例会先 warm-up；不要把首次时间当吞吐。

**要掌握的术语。**

- H2D（host-to-device）：`cpu_tensor.to("cuda")`。
- D2H（device-to-host）：`gpu_tensor.cpu()`；它常是同步点。
- 异步：CPU 把 CUDA kernel 放入队列后会立即返回；测 GPU 耗时前需 `torch.cuda.synchronize()`。

**练习。** 把矩阵尺寸改为 `2048`，比较 warm-up 前后的时间；再故意删除同步，观察不可信的计时。

**验收不变量。** 除最终 token id 外，任何大张量都不应为了生成过程而回到 CPU。

## 01. 一个请求的 prefill、decode 与连续 KV cache ✅

**要解决的问题。** 若第 n 个 token 都重新计算前 n 个 token 的 K/V，生成长度越长，重复计算越多。注意力中，当前 query 只需要“过去 token 的 K/V”和“当前 token 的 Q/K/V”。因此，提示词的一次完整计算称为 **prefill**；每个新 token 的单步计算称为 **decode**。

**本章模型。** `ToyAttentionLM` 只有一层、一个 attention head。它不是会说话的模型，而是一个可精确验证的 decoder。对输入隐藏状态 `x`：

```text
Q = xWq,  K = xWk,  V = xWv
attention(Q, K, V) = softmax(QKᵀ / √H + causal_mask)V
```

`KVCache.key` 与 `.value` 的布局均为 `[batch, token_position, hidden_size]`。真实模型还会有 layer/head 维度；先省略它们，避免掩盖核心事实。

**按步骤执行。**

1. 阅读 `mini_vllm/toy_model.py`，先画出 `input_ids -> hidden -> Q/K/V -> logits` 的形状。
2. 看 `forward(input_ids, cache=None)`：没有 cache 且 token 数大于一是 prefill；有 cache 且 token 数为一是 decode。
3. 看 `mini_vllm/generation.py`。`greedy_generate_full` 是正确但低效的基线；`greedy_generate_cached` 是优化版。
4. 运行：

```powershell
python lessons/01_prefill_decode.py
```

两行生成 token 序列必须完全一致，最后打印 `PASS`。再运行通用测试；`test_decode_logits_equal_full_context_logits` 会比较同一上下文下两种路径的浮点 logits。

**最重要的不变量。** 对任意相同的 prompt 与下一 token：

```text
model(prompt + next_token).last_logits
== model(next_token, cache=model(prompt).cache).last_logits
```

这就是以后所有 KV cache 优化都不能破坏的正确性定义。

**常见错误。**

- decode 的位置编码从 `0` 重启：会让缓存路径与完整路径不一致。
- cache 缺少 causal mask：同一批 prefill 的 token 会偷看未来。
- 只缓存 K，不缓存 V：attention 仍无法得到历史上下文。
- 在每一步调用 `.item()`：会产生 GPU→CPU 同步；本章为了展示结果允许它，服务层会限制这种边界。

## 02. 将模型执行收敛为 ModelRunner ✅

**要解决的问题。** 若调度器直接调用模型、复制张量并采样，后续换模型/换 cache 布局会牵动所有代码。`ModelRunner` 应成为唯一知道“GPU 上如何执行”的组件。

**本章交付。**

```text
CPU scheduler -- ExecutionRequest --> ModelRunner(GPU) --> sampled token IDs
```

实现文件：[`mini_vllm/runner.py`](../mini_vllm/runner.py)、[`lessons/02_model_runner.py`](../lessons/02_model_runner.py)。它只有 `ReferenceModelRunner`，不导入 pool。运行：

```powershell
python lessons/02_model_runner.py
```

**实现步骤。**

1. `ReferenceModelRunner.prefill()` 把 CPU 的 prompt token IDs 显式拷到 GPU，调用第 01 章模型。
2. 它在 GPU `argmax`，只将 `int token_id` 交回 CPU；连续 `KVCache` 仍留在 GPU。
3. `decode()` 接收上一步 token 和 cache，复用第 01 章已有的 `model(input, cache)` 接口。
4. 此时不解决 slot 所有权，也不做异构 batch；那些是第 03/05 章的新问题。

**验收。** `test_reference_runner_preserves_the_chapter_one_result` 会将 Runner 的两次输出与第 01 章的完整重算比较。测试通过表示：CPU 侧的 Python `list[int]` 可以驱动 GPU forward，而连续 K/V 从未回到 CPU。

## 03. 把连续 KV cache 变成可管理的资源 ✅

**要解决的问题。** 第 01 章中每次 `torch.cat` 都分配新张量，适合解释但不适合服务。服务需要预分配容量，并只写入新 token 的槽位。

**本章交付。** `KVCachePool` 在 GPU 预分配 `[sequences, max_tokens, hidden_size]` 的 K 和 V。为突出 slot 生命周期，toy 模型省略了 layer/head 维度；真实模型的每层都需要相同管理。`acquire(owner_id)`、`write(...)`、`read(...)`、`release(...)` 明确了所有权。`PooledAttentionExecutor` 与 `PooledModelRunner` 是**新增适配器**：第 01/02 章源码不因此改变。运行：

```powershell
python lessons/03_cache_pool.py
```

**先做纸面计算。** 单个 sequence 的连续 cache（忽略 batch 对齐）近似占用：

```text
2 × num_layers × sequence_length × hidden_size × dtype_bytes
```

因子 2 来自 K 与 V。对 fp16/bf16，`dtype_bytes = 2`。把你的层数、hidden size、上下文长度代入，先估算 8 GiB 显存能容纳多少并发请求。

**实现步骤。**

1. 定义 slot 所有权：一个 `sequence_id` 在未释放前独占其槽位。
2. 先编写越界、重复释放、写入位置不连续的失败测试。
3. 在 `PooledModelRunner.prefill()` 中写入 prompt 的 K/V，再在 `decode()` 中原地写入上一步生成 token 的 K/V。
4. `PooledAttentionExecutor.forward()` 只读取当前 batch 所需的最长历史；对较短行应用 validity mask，令异构长度仍能组成 GPU batch。
5. 将结果与第 01 章的连续 cache logits 比较。

**验收。** cache 地址在 decode 时不变；满池时获得确定的 `OutOfCacheSlots`；释放后 slot 可被新 sequence 重用且不会读到旧数据。

## 04. 用请求状态机表达服务语义 ✅

**要解决的问题。** 模型只接受 tensor，但服务面对的是独立生命周期的请求：排队、预填充、逐 token 生成、完成、取消、失败。把这些状态混进 scheduler 会让测试极难写。

**本章交付。** `Request` 与明确转移：

```text
WAITING -> RUNNING -> FINISHED
              |          
              +-> CANCELLED / FAILED
```

**字段最小集。** `request_id`、prompt token、已生成 token、最大新 token 数、eos token、状态、cache slot、错误信息。本课程一个 request 对应一个模型上下文；beam search 与时间指标不在本章引入。

**实现步骤。** 先以纯 Python 状态机实现，不触碰 GPU。用 table-driven test 覆盖合法与非法转移。只有状态机稳定后才把它交给 scheduler。

运行 `python lessons/04_request_state.py`。**验收。** 生成到 `max_new_tokens` 或 EOS 后恰好一次进入 `FINISHED`；取消后不得再接收 token；终态不得回到 `RUNNING`。`test_last_generated_token_is_not_yet_in_cache` 特别验证一个容易混淆的时刻：模型预测出来的 newest token 要等到下一次 decode 才会写进 K/V。

## 05. 连续批处理调度器 ✅

**要解决的问题。** 静态 batch 必须等待最长请求完成，延迟和吞吐都很差。连续批处理在每个 decode step 完成后移除结束请求，并让新请求加入后续 step。

**本章交付。** `Scheduler.schedule()` 返回一次 execution plan：优先让已运行请求各做一个 decode，再在剩余 token budget 内接纳 FIFO waiting 请求做 prefill。

**单步时序。**

```text
t0: A(prefill), B(prefill)
t1: A(decode), B(decode), C(prefill)
t2: A(finished), B(decode), C(decode), D(prefill)
```

**实现步骤。**

1. 以 `max_batched_tokens` 作为唯一准入预算；不要先引入多个难以解释的阈值。
2. FIFO 接纳 waiting 请求，保证可复现。
3. 在 Runner 返回 token 后更新状态、释放完成 sequence 的 cache slot。
4. 同一个调度 step 中不允许 sequence 既 prefill 又 decode。
5. 当前 `Engine` 为了清楚地支持不同 prompt 长度，逐请求 prefill；已有 running 请求使用一次异构长度的 GPU decode batch。这是一个可验证的中间实现，而不是把“批处理”偷换成循环。
6. 入队时立即拒绝“单步 token budget 放不下”或“KV slot 容量放不下”的请求，避免一个永远不会被调度的请求静默卡住队列。

运行 `python lessons/05_continuous_batching.py`。**验收。** `test_continuous_batching_matches_single_request_reference` 用两个不同 prompt/生成长度请求，断言每个输出与“单独生成”一致，并断言结束后 GPU slot 使用量回到零。

## 06a. Block manager：分页 cache 的元数据 ✅

**要解决的问题。** 连续 cache 按“最大长度 × 最大并发”预留空间，产生内部碎片。分页思路是把 token 空间切成固定大小 block，每个 sequence 只有逻辑 block table，物理 block 可以复用。

**本章交付。** `BlockManager` 与逻辑 block table 元数据。每个物理 block 固定容纳 `block_size` 个 token；GPU paged K/V gather 留给新增的 06b，当前不声称已有 `PagedKVCache`。

**核心映射。**

```text
logical_block = position // block_size
offset        = position % block_size
physical_block = block_table[logical_block]
```

**实现步骤。**

1. 先以 CPU list 实现 free-block 队列和逻辑映射；为零 block、跨 block 写、重复 free 写测试。
2. 运行 `python lessons/06_block_manager.py`，看 token 位置怎样变成 `(physical_block, offset)`。
3. 当前的 [`BlockManager`](../mini_vllm/block_manager.py) 已管理 free list、逻辑表、越界和 release，并有独立测试。主 Engine 仍使用连续 slot pool；将 table 真正接入 GPU gather 是下一层优化，不应伪装成已经完成的 PagedAttention kernel。
4. 以后接入时，要对比连续 cache 与分页 cache logits；二者必须匹配。

**验收。** 断言实际分配 block 数等于 `ceil(sequence_length / block_size)`；序列结束后所有 block 回到 free list；任意两个活跃 sequence 没有同一个物理 block。

## 07a. 把引擎变为最小 HTTP 推理服务 ✅

**要解决的问题。** 一个正确的 loop 仍不是服务：客户端需要创建请求、得到逐 token 输出、取消请求、查询健康状态；运维需要观测队列和 GPU 使用。

**本章交付。** Python 标准库 `ThreadingHTTPServer` 和线程安全的 `InferenceService` 门面。每个 HTTP handler 推进一次或多次 `Engine.step()`；后台 worker、ASGI 和真实 SSE 留给 07b。

- `POST /v1/completions`：创建请求；`stream=true` 时返回 NDJSON。
- `POST /v1/requests/{id}/cancel`：请求取消。
- `GET /health`：模型/GPU 就绪。
- `GET /metrics`：waiting/running 数和连续 cache slot 使用量。

**实现步骤。** HTTP handler 只负责校验和入队；只有 Engine 调用 Scheduler；只有 Runner 使用 GPU。用内存队列替代 Redis 等外部依赖。当前实现位于 [`mini_vllm/service.py`](../mini_vllm/service.py)，使用 Python 标准库，避免为了教程引入额外 Web 框架。

```powershell
python lessons/07_service.py
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod -Method Post http://127.0.0.1:8000/v1/completions -ContentType application/json -Body '{"prompt_token_ids":[1,2],"max_new_tokens":3}'
```

**验收。** 两个并发流式请求都按序收到 token；取消请求不会继续输出；健康检查不触发一次模型执行；指标在请求后变化并可解释。

## 08a. 从“正确”走向“可复现测量” ✅

**要解决的问题。** 性能优化不能凭直觉。消费级卡上尤其需要知道受限于算力、显存、同步还是 CPU 排队。

**本章交付。** `benchmarks/decode.py`，比较完整重算、连续 cache reference 路径与 Engine cache-pool 路径。静态/连续 batch 对比、block cache 对比、p50/p95 指标留给新增的 08b。

**测量规则。** warm-up 后再取样；使用 CUDA events；报告中位 GPU 时间。08b 再加入首 token 延迟、吞吐、峰值显存与 p50/p95。

运行 `python lessons/08_profiling.py`。脚本会先断言三条路径的输出相同，再分别报告完整重算、连续 cache 和 Engine cache-pool 的 CUDA event 中位数。小模型上 Engine 路径可能更慢：Python 调度、slot gather、每 token D2H 比节省的数学计算更显著。这不是失败，而是下一次优化的证据。

**验收。** 每项性能结论都有脚本和参数可复跑。若优化改变输出，优先修正确性测试，不能用“更快”掩盖差异。

## 09–13a. 可选的真实 Hugging Face 模型轨道 ✅

完成 00–08 后，不要把 toy 模型替换掉；转而按[真实 Hugging Face 模型扩展](hf-track.md)新增一条并行的兼容性轨道。它使用 `Qwen/Qwen2.5-0.5B-Instruct`，把第 01–03 章的 cache 概念迁移到真实的 24 层 GQA 模型，但不伪称已经拥有 Qwen 的 PagedAttention kernel。

| 章节 | 新增能力 | 验收 oracle |
| --- | --- | --- |
| 09 | 原生 chat template + `generate()` | Qwen 在 `cuda:0` 生成可复现 greedy token IDs |
| 10 | 显式 prefill / native cache / decode | FP32 下 cached logits 与整段前向严格一致 |
| 11 | `HuggingFaceReferenceRunner` | 其 greedy token IDs 等于 native `generate()` |
| 12 | GQA K/V 形状和字节估算 | 真实 cache 为 24 层、2 个 KV heads、head dim 64 |
| 13a | 独立 loopback HTTP 服务 | `/health` 与 `/v1/chat/completions` 使用真实模型通过 |

这五章的背景、逐步代码阅读、BF16/FP32 取舍、命令与常见兼容性问题都在该扩展文档中。关键边界是：第 10 章选择 FP32 做数值 oracle；第 09、11–13 选择 BF16 作为 8 GiB 单卡服务的默认值。后续把真实模型接入 Scheduler、再实现 layer-aware paged K/V execution，应作为 12b/13b 新章节，不能倒灌改写前置章节。

## 最终集成验收（课程完成条件）

满足以下全部条件才把项目称为“一个正确的 mini 推理框架”：

1. 单请求的 cached decode 与完整重算的 token/logits 一致。
2. 多请求连续批处理的每个输出都与单请求参考实现一致。
3. 连续 cache slot 不泄漏、不双重分配，取消与完成都会释放资源；06a 的 block table 也有独立的释放测试。
4. GPU 只在 Runner 内被使用；调度与 HTTP 层不持有 CUDA tensor。
5. 服务门面可为多个请求分别收集 token；HTTP health/completion、取消和 slot 指标有测试。07b 再把它扩展为后台流式服务。
6. 性能结果可由 benchmark 命令复现，并说明对应硬件、模型和参数。
7. （可选真实模型轨道）Qwen 的 FP32 cache parity 与 native-generate token parity 均通过；其 HTTP baseline 的功能范围明确标为单请求，而非连续批处理。

到此得到的结构会对应 vLLM 的主要概念：`Engine`、`Scheduler`、`ModelRunner`、`Sequence/Request` 和 `BlockManager`。差别是我们先用可读的 PyTorch 实现验证每项不变量，而 vLLM 用高度优化的 CUDA kernel、分布式执行和更丰富的策略把同一类思想扩展到生产规模。
