---
layout: default
title: mini-vLLM 教程
---

# mini-vLLM：从 CPU/GPU 协作到推理服务

> 一个教学优先、单卡可运行、每一步都可验证的 mini 推理框架。

这不是 vLLM 的缩小复刻，而是一条把它的关键问题逐个还原出来的学习路线：为什么 KV cache 必须留在 GPU？为什么请求生命周期不能混进模型代码？为什么连续批处理、块映射和服务接口要分阶段引入？

## 你会得到什么

- 能直观看到 CPU ↔ GPU 边界、同步点与计时陷阱。
- 一个可精确对比 full recompute 与 cached decode 的 toy decoder。
- 具有 request 状态机、连续批处理、GPU 连续 KV pool、block 元数据和 HTTP 门面的 mini 推理引擎。
- 一条独立的真实模型兼容轨道：`Qwen/Qwen2.5-0.5B-Instruct` 的 native cache、GQA cache 观察、greedy token oracle 与最小 HTTP 服务。
- 可重复运行的单元测试、真实模型集成测试和性能基准。

## 学习地图

| 阶段 | 解决的问题 | 可验证交付物 |
| --- | --- | --- |
| 00–02 | CPU/GPU 边界与单请求正确性 | H2D/D2H 观察、toy prefill/decode parity、reference runner |
| 03–05 | KV cache 的资源化与并发调度 | 固定 slot pool、request 状态机、连续批处理 engine |
| 06a–08a | 分页思想、服务化与测量 | block table、HTTP 门面、可复现 benchmark |
| 09–11 | 真实模型正确性桥梁 | Qwen native oracle、FP32 cache parity、Transformers adapter |
| 12–13a | 真实 GQA cache 与服务边界 | KV heads 显存估算、独立真实模型 HTTP baseline |

每一阶段都是“问题 → 最小接口 → 可运行实现 → 测试/观测 → 与 vLLM 的对应关系”。后面的优化只能新增模块或 adapter，不能回头污染已经学完的基础实现。

## 从这里开始

1. 阅读[完整学习指南](learning-guide.md)，按 00 → 08 完成 toy 路径。
2. 阅读[真实 Hugging Face 模型扩展](hf-track.md)，再开始 09 → 13a。
3. 遇到“为什么不直接改旧代码”时，查阅[课程增量约定](incremental-contract.md)。

## 当前边界

项目已经具备“正确、可观察的单卡教学引擎”，但它**不是生产 vLLM**：block manager 目前是经过验证的元数据层，不包含 Qwen 的 paged-attention kernel；真实模型 HTTP 路径目前是单请求 greedy baseline，不假称已有连续批处理。后续的 06b、12b、13b 将在前述 correctness oracle 之上继续扩展。

## 在线阅读

这个目录可直接作为 GitHub Pages 的发布源。将仓库推送至 GitHub 后，在仓库 **Settings → Pages** 中选择 **Deploy from a branch**，发布分支选 `main`、目录选 `/docs`。站点地址通常为：

```text
https://<GitHub 用户名>.github.io/<仓库名>/
```

完整命令与注意事项请看仓库根目录的 README「发布到 GitHub Pages」一节。
