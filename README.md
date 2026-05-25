# SpeculativeExecutor

用分支预测的方式加速 Agent 工具调用 — 基于 DeepSeek Prefix Continuation API 在 BFCL v4 基准上的实验。

> 实现和数据：[用分支预测加速 Agent 工具调用](https://zhuanlan.zhihu.com/p/2041639052571566901)

## 核心结果（BFCL v4, DeepSeek V4 Flash）

| 类别 | 样本 | Main 准确率 | Spec 准确率 | 加速比 | Token 占比 |
|---|---|---|---|---|---|
| simple_python | 400 | 92.2% | 66.0% | 1.8x | 27% |
| multi_turn_base | 200 | 2.5% | 91.0% | 11.1x | 36% |
| irrelevance | 240 | 70.0% | 44.6% | 3.9x | 18% |

- **单轮**：Main 在参数精度上领先，Spec 受限于缺少参数描述信息
- **多轮**：Main 陷入探索循环（pwd/ls/状态查询占 37% 调用），Spec 直接命中目标工具
- **无关工具**：两者都面临 false positive 问题，Spec 更差（55%）

## 快速开始

```bash
git clone https://github.com/wefio/SpeculativeExecutor.git
cd SpeculativeExecutor
uv sync

# 配置 API Key
cp .env.example .env
# 编辑 .env 填入你的 DeepSeek API Key

# 运行实验
uv run experiment.py --category simple_python --max-samples 10
uv run experiment.py --category multi_turn_base --max-samples 5
uv run experiment.py --category irrelevance --max-samples 10
uv run experiment.py --category all --concurrency 10
```

## 工作原理

```
用户问题 ──┬──> Main 模型 (chat) ──────────> tool_call (3s)
           │
           └──> Prefix 续写 (prefix_complete) ──> tool_call (1s)
                                                      │
                                              提前执行，缓存结果
                                                      │
                                              Main 完成时命中缓存
```

## 项目结构

```
provider.py         DeepSeek API 封装 (chat, prefix_complete, fim_complete)
speculative/        投机执行器，prefix 预测工具调用
tools.py            工具注册中心，支持 BFCL 工具
agent.py            Agent 循环，支持投机模式
experiment.py       实验入口，加载/执行/评估/输出
benchmark_bfcl.py   BFCL 数据加载 + AST 评估库
experiments/        三类完整实验数据 (JSON)
```

## 引用

```bibtex
@misc{speculative-executor,
  title   = {用分支预测的方式加速 Agent 工具调用},
  author  = {wefio},
  year    = {2026},
  url     = {https://github.com/wefio/SpeculativeExecutor},
}
```

## License

MIT
