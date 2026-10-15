# mini-vLLM：从 CPU/GPU 协作到推理服务

> 用一张消费级 GPU，从零实现一个**正确、可观察、可验证**的 mini 推理框架。

`mini-vLLM` 是一套教学优先的增量式教程与代码库。它不试图把生产版 vLLM 的 CUDA kernel、分布式并行和全部调度策略压缩到几百行；它要做的是把一个推理服务真正需要解决的问题逐个拆开，并在每一步留下可以运行、可以测试、可以和前一步对照的实现。

**在线教程入口（GitHub Pages 准备就绪）：** [`docs/index.md`](docs/index.md) · [完整学习指南](docs/learning-guide.md) · [真实模型扩展](docs/hf-track.md)

---

## 这个项目现在能做什么？

| 能力 | 当前实现 | 你能观察到什么 |
| --- | --- | --- |
| CPU / GPU 边界 | 显式 H2D、GPU 计算、D2H token 返回 | 为什么 CUDA 计时必须同步；为什么不应把大张量搬回 CPU |
| 单请求生成 | 一层 one-head `ToyAttentionLM` | prefill 与 decode 的计算差异、causal mask 与位置增长 |
| KV cache 正确性 | 连续 cache reference path | cached decode 的 logits / token 与完整重算一致 |
| GPU cache 资源 | 预分配 `KVCachePool`、slot 所有权 | 分配、写入、释放、满池拒绝与重用都可测试 |
| 多请求服务语义 | `Request` 状态机、`Scheduler`、`Engine` | waiting / running / finished / cancelled 如何安全迁移 |
| 连续批处理 | decode 优先、FIFO prefill、token budget | 新请求怎样在旧请求 decode 的间隙加入 |
| 分页思想 | `BlockManager` 逻辑 block table | token position 如何映射为 `(physical_block, offset)` |
| 最小服务与测量 | 标准库 HTTP 门面、CUDA event benchmark | completion / cancellation / metrics，以及性能瓶颈在哪里 |
| 真实模型桥接 | Qwen 0.5B、Transformers native cache、GQA 观察 | toy 概念如何迁移到真实 tokenizer、RoPE、GQA 与多层 K/V |

项目的核心不是“能输出一句话”，而是每项优化都有一个**不改变语义的证明对象**：toy 路径与真实模型路径都保留 reference/oracle，先比 token 或 logits，再谈更快的实现。

## 它刻意不做什么

为了让学习路径可信、可在 8 GiB 消费级显卡上运行，下面这些能力被明确推迟，而不是半实现后混进主线：

| 尚未实现 | 为什么延后 |
| --- | --- |
| 真正的 paged-attention CUDA gather / kernel | 06a 先验证 block 映射和生命周期；kernel 应在 06b / 12b 作为独立优化引入 |
| 将 Qwen 接入连续批处理 Engine | 真实模型需要先定义异构 batch、native cache 所有权与 EOS 语义；留到 13b |
| 分布式推理、量化 kernel、CUDA Graph、投机采样 | 它们会遮蔽本教程要先建立的单卡 correctness invariant |
| 生产级流式协议与后台 worker | 07a 先固定 HTTP 边界；异步 serving 是后续 07b 的独立问题 |

这比“一份看似功能很多、但无法说明哪里正确”的代码更适合跟读和继续扩展。

---

## 为什么要分阶段，而不是直接实现一个最终框架？

推理引擎中的问题有依赖顺序：如果你还不能证明单请求 cached decode 与完整重算一致，就没有资格讨论 batch；如果不明确请求何时结束、cache 归谁所有，就不能安全谈 block 复用；如果没有原生模型 oracle，就不能判断自定义真实模型路径是在加速还是在悄悄算错。

```text
正确性与边界
00 device ──> 01 toy KV cache ──> 02 reference runner
                                      │
资源与并发                          ▼
03 GPU pool ──> 04 request state ──> 05 scheduler / engine
                                      │
服务与证据                          ▼
06a block metadata ──> 07a HTTP ──> 08a benchmark
                                      │
真实模型兼容                        ▼
09 HF oracle ──> 10 cache parity ──> 11 adapter ──> 12 GQA ──> 13a HTTP
```

每一章遵守同一个节奏：

1. **提出一个只属于这一章的问题**，例如“谁拥有 GPU slot？”
2. **引入最小接口**，不提前把 scheduler、HTTP、pool 塞回 toy model。
3. **运行一个最小实现**，让数据流和张量形状可见。
4. **写下不变量并验证**，通常是 token/logits 等价、slot 不泄漏或状态转换合法。
5. **标出与 vLLM 的对应关系和刻意缺口**，防止把教学实现误当生产实现。

这份[增量约定](docs/incremental-contract.md)是课程的架构契约：后续章节优先新增 adapter，而不是重构已经学完的文件。

---

## 完成后你会读懂的整体结构

```text
                         CPU
HTTP / tokenizer / request state / scheduler / engine
                 │                       │
                 │ execution plan        │ token events
                 ▼                       ▼
            ModelRunner ─────────────> InferenceService
                 │  H2D token IDs
                 ▼
                         GPU
ToyAttentionLM / Qwen model + K/V cache
                 │  GPU argmax
                 ▼
            only a small token ID crosses D2H
```

toy 主线中的 `PooledModelRunner` 管理预分配连续 KV pool；真实模型扩展中的 `HuggingFaceReferenceRunner` 暂时把 `past_key_values` 交给 Transformers。二者共享同一个思想：**CPU 组织工作，GPU 持有大张量，跨边界只传必要的小数据。**

---

## 课程地图

所有章节都已有代码、lesson 与测试。推荐严格按序进行。

| 阶段 | 做了什么 | 关键文件 | 验收方式 | 对应的推理系统概念 |
| --- | --- | --- | --- | --- |
| 00 | 看见 CPU / GPU 边界与 CUDA 同步 | `device.py` | 观察 H2D、GPU、D2H device | host-device orchestration |
| 01 | 写一个可精确验证的 causal decoder | `toy_model.py`、`generation.py` | full 与 cached logits / token 一致 | prefill、decode、KV cache |
| 02 | 收敛 GPU 调用边界 | `runner.py` | runner 保留第 01 章结果 | ModelRunner |
| 03 | 将 cache 变成具所有权的 GPU 资源 | `cache.py`、`pooled_runner.py` | slot 地址稳定、释放后可重用 | KV cache manager |
| 04 | 将服务语义写成纯 Python 状态机 | `requests.py` | 非法状态转移失败 | sequence/request lifecycle |
| 05 | 调度多请求并做连续批处理 | `scheduler.py`、`engine.py` | 多请求等价于各自单独运行 | scheduler、continuous batching |
| 06a | 先验证分页映射，而非假装有 kernel | `block_manager.py` | logical block 与 physical block 映射正确 | block table、PagedAttention 思想 |
| 07a | 用最小 HTTP 门面暴露引擎 | `service.py` | completion、取消、health、metrics | serving boundary |
| 08a | 建立可重复的性能证据 | `benchmarks/decode.py` | CUDA event + 输出等价检查 | profiling discipline |
| 09 | 用原生 Qwen `generate()` 建立 oracle | `hf_backend.py` | 固定 greedy token IDs | tokenizer、chat template |
| 10 | 严格验证真实模型 cache | `hf_runner.py` | FP32 cached/full logits 一致 | native `past_key_values` |
| 11 | 将真实模型收敛到 prefill/decode adapter | `hf_runner.py` | adapter token IDs = native generate | execution adapter |
| 12 | 观察真实 GQA K/V 布局与字节数 | `hf_cache.py` | 24 layer / 2 KV heads / head dim 64 | layer-aware cache planning |
| 13a | 暴露单请求真实模型 HTTP baseline | `hf_service.py` | `/health` 与 chat completion | real-model serving boundary |

逐步解释、练习、常见错误和每章的限制，请从[完整学习指南](docs/learning-guide.md)进入；09–13a 的真实模型细节见[HF 扩展路线](docs/hf-track.md)。

---

## 快速开始

### 运行环境

已验证环境：Windows、Python 3.12、PyTorch 2.5.1+cu121、CUDA、RTX 4060 Laptop GPU（8 GiB）。toy 路径不需要下载大模型，也不需要安装 vLLM。

建议先按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择与你 CUDA 驱动匹配的 wheel，然后在仓库根目录运行 lesson。最小检查：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m unittest discover -s tests -v
```

### 先完成 toy 路径（00–08）

```powershell
python lessons/00_cpu_gpu.py
python lessons/01_prefill_decode.py
python lessons/02_model_runner.py
python lessons/03_cache_pool.py
python lessons/04_request_state.py
python lessons/05_continuous_batching.py
python lessons/06_block_manager.py
python lessons/07_service.py
python lessons/08_profiling.py
```

每完成一章都运行通用回归：

```powershell
python -m unittest discover -s tests -v
```

### 再开启真实 Qwen 路径（09–13a）

默认真实模型是 [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)。它的 0.49B 参数、GQA 架构和中文能力适合作为 8 GiB 单卡上的真实模型样本；模型的具体架构与使用方式以[官方模型卡](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)和 [Transformers Qwen2 文档](https://huggingface.co/docs/transformers/model_doc/qwen2)为准。

```powershell
pip install -e ".[hf]"

python lessons/09_hf_baseline.py
python lessons/10_hf_cache_parity.py
python lessons/11_hf_runner.py
python lessons/12_hf_gqa_layout.py
python lessons/13_hf_service.py
```

第 09、11–13a 默认 BF16，控制显存占用；第 10 章故意以 FP32 做严格 logits oracle。cached decode 与 full forward 的 attention 矩阵形状不同，BF16 的累加舍入会让“逐元素严格相等”失去意义；FP32 能验证 cache 语义本身，而不是误把数值格式误差当 bug。

模型已下载后，额外运行真实模型集成测试：

```powershell
$env:MINI_VLLM_RUN_HF=1
python -m unittest tests.test_hf_integration -v
```

---

## 如何确认你没有“只是跑通了”？

| 检查层次 | 要证明的事情 | 命令 / 证据 |
| --- | --- | --- |
| toy 数学正确性 | cached decode 没有改变模型语义 | `test_decode_logits_equal_full_context_logits` |
| 引擎正确性 | 连续批处理和单请求参考结果一致 | `test_continuous_batching_matches_single_request_reference` |
| 资源正确性 | slot / block 会释放，所有权不会交叉 | `test_pool_refuses_release_by_wrong_owner`、block tests |
| 边界正确性 | 前置章节不偷用后续优化 | `test_chapter_boundaries.py` |
| 真实模型正确性 | native cache 与整段前向、native generate 一致 | `MINI_VLLM_RUN_HF=1` 的 2 项测试 |
| 服务正确性 | health 与 completion 使用正确边界 | toy HTTP tests + 13a loopback lesson |

当前已在本机完成过：24 项普通测试通过；真实 Qwen 的两项 opt-in 集成测试通过；真实 HTTP `/health` 与 `/v1/chat/completions` 通过。普通测试刻意不下载模型，真实模型测试也必须由环境变量显式开启。

---

## 发布为 GitHub Pages 在线教程

本仓库已经准备好了 GitHub Pages 源目录：

```text
docs/
├── index.md                  # 在线教程首页
├── _config.yml               # Jekyll 标题、描述与主题
├── learning-guide.md         # toy 主线的逐章教程
├── hf-track.md               # 真实 Qwen 扩展路线
└── incremental-contract.md   # 增量架构契约
```

推荐使用 **Deploy from a branch → `main` → `/docs`**：它没有前端构建步骤，正适合当前 Markdown 教程。GitHub 官方也建议不需要自定义构建时直接从分支和 `/docs` 目录发布。[GitHub Pages 发布源说明](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)

### 第一次发布

1. 在 GitHub 创建一个普通仓库，例如 `mini_vllm`，并把当前项目推送到默认分支 `main`。
2. 进入仓库 **Settings → Pages**。
3. 在 **Build and deployment** 中选择 **Deploy from a branch**。
4. 选择分支 `main` 和目录 `/docs`，保存。
5. 等待 Pages 部署完成后，访问：

   ```text
   https://<你的 GitHub 用户名>.github.io/mini_vllm/
   ```

在此模式下，后续只要推送 `docs/` 中的 Markdown，教程站点就会随之更新。GitHub 也支持改用 GitHub Actions；当未来加入 MkDocs、Docusaurus、生成 API 文档或复杂前端构建时，再切换到 Actions 更合适。[GitHub 官方说明](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)

如果希望站点直接使用 `https://<用户名>.github.io/` 而不带 `/mini_vllm/`，需要把仓库命名为 `<用户名>.github.io`；这是 GitHub 的“用户站点”命名规则。[GitHub Pages quickstart](https://docs.github.com/en/pages/quickstart)

> 发布前检查：GitHub Pages 对外公开内容；不要把 Hugging Face token、缓存路径、测试日志或任何密钥提交进仓库。

---

## 下一步建议

完成 13a 后，最自然的延伸不是立刻优化 kernel，而是继续遵循相同的证明顺序：

1. **06b / 12b：** 将 block table 接到一层真实 K/V gather，并先与连续 cache 做 logits parity。
2. **13b：** 设计 HF runner 与 Scheduler 的所有权模型，支持真实模型的连续批处理。
3. **07b / 08b：** 添加后台 worker、流式协议、TTFT、decode token/s、峰值显存和分位数指标。
4. 只有在这些证据都具备后，再研究 CUDA Graph、量化、flash attention 或更复杂的 vLLM 调度策略。

这样，每一次“更快”都有前一步的“仍然正确”作为证据。
