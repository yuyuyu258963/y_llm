# 课程增量约定

本项目不是一次性展示“最终文件”，而是按主线不断新增可验证能力。下面的依赖表是课程契约；违反它的改动需要拆到更晚的章节，不能回写到已完成的基础章节。

| 章节 | 冻结的最小能力 | 允许新增的模块 | 禁止的依赖 |
| --- | --- | --- | --- |
| 00 | CPU/GPU 边界 | `device.py` | 模型、请求、cache pool |
| 01 | 单请求连续 KV cache | `toy_model.py`、`generation.py` | pool、scheduler、HTTP |
| 02 | 模型调用边界 | `runner.py` 的 `ReferenceModelRunner` | `KVCachePool`、请求状态机 |
| 03 | 预分配连续 pool | `cache.py`、`pooled_attention.py`、`pooled_runner.py` | scheduler、HTTP |
| 04 | 请求生命周期 | `requests.py` | CUDA tensor、HTTP |
| 05 | 调度与 Engine | `scheduler.py`、`engine.py` | block table、HTTP transport |
| 06a | 分页元数据 | `block_manager.py` | 改写连续 pool 主路径 |
| 07a | 服务门面与 HTTP transport | `service.py` | 改写 Engine 调度语义 |
| 08a | 基准与测量纪律 | `benchmarks/` | 为了跑分修改语义 |
| 09 | 原生 HF oracle | `hf_backend.py` | 回写或替换 00-08 的 toy 基线 |
| 10-11 | 原生 cache parity / adapter | `hf_runner.py` | 修改 toy Runner、Scheduler 或 Engine |
| 12 | 真实 GQA cache 观察 | `hf_cache.py` | 假称已实现 Qwen PagedAttention |
| 13a | 独立 HF HTTP 基线 | `hf_service.py` | 将单请求路径伪装成连续批处理服务 |

## 修改规则

1. 已完成章节的 public API 和测试先视为冻结。新概念优先新建模块或适配器。
2. 新实现必须保留旧实现作为 reference，并以 token/logits 一致性测试证明替换正确。
3. 一个章节只能依赖本章和更早章节；`tests/test_chapter_boundaries.py` 自动检查第 01、02 章没有导入后续 pool 优化。
4. 只有在某项性能或正确性证据要求时，才修改较早章节；修改必须同时说明不变量为何保持。
5. 06a 的 block table、07a 的标准库服务、08a 的基础 benchmark 都是有意保留的阶段性终点。Paged KV gather、后台 Engine loop 和分位数指标分别应作为 06b、07b、08b 的**新增章节**。
6. Hugging Face 接入从 09 章开始且为可选路径；它必须把 native Transformers `generate()` 保留为 oracle，不能替换 toy reference。第 10 章为严格 logits 对比选择 FP32；第 09、11-13 默认 BF16 只是在显存和数值精度间的教学取舍。

## 当前学习顺序

```text
00 device boundary
  -> 01 reference decoder + growing KV cache
  -> 02 reference runner
  -> 03 fixed-slot pool + pooled runner
  -> 04 request state
  -> 05a scheduler, 05b engine integration
  -> 06a block table, 06b paged KV execution
  -> 07a service transport, 07b asynchronous serving and metrics
  -> 08a reproducible benchmark, 08b optimization study
  -> 09 HF baseline -> 10 cache parity -> 11 HF runner -> 12 real GQA blocks -> 13 real serving
```

这条路径确保后面每一步是在前面正确代码上新增一层，而不是回头把“答案”塞进已经学过的章节。
