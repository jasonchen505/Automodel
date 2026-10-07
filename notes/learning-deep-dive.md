# NeMo AutoModel 深入学习文档

> 研读对象：`jasonchen505/Automodel`（fork 自 `NVIDIA-NeMo/Automodel`），研读时间 2026-10-07。
> 这是一份个人学习笔记，放在 fork 的 `notes/learning` 分支上，不进入 upstream PR。

---

## 1. 一句话定位

NeMo AutoModel 是 **NVIDIA NeMo Framework 生态下的 PyTorch DTensor-native SPMD 开源训练库**，
面向 LLM / VLM / diffusion / retrieval 模型的训练与微调。两个核心卖点：

1. **DTensor-native SPMD**：同一份训练脚本，改 device mesh 配置就能从 1 卡扩到上千卡——
   并行策略是运行时布局选择，不是代码分支；
2. **Day-0 Hugging Face 支持**：新模型上 Hub 当天就能训，无需 checkpoint 转换，checkpoint 全程保持 HF 原生格式。

与周边项目的关系：
- **NeMo Framework**：隶属关系；RLHF/DPO/PPO 明确指向兄弟项目 **NeMo-RL**（`tutorials/README.md`）；
- **Megatron-LM**：不是基于它重写——`components/distributed/megatron_fsdp.py` 把 MegatronFSDP 做成与
  FSDP2、DDP 并列的三种分布式策略之一；
- **torchtitan**：无依赖，只借鉴风格（`moe/parallelizer.py` 的 "TorchTitan-style per-op selective
  activation checkpointing" 等）。

## 2. 核心技术：DTensor-native SPMD

**SPMD** = Single Program, Multiple Data。技术底座是 PyTorch 原生的 `DeviceMesh` + placements
（`Shard`/`Replicate`）：模型/优化器状态按 mesh 切分，模型代码保持纯 PyTorch，并行逻辑活在配置里。
README 原话 TL;DR：*"SPMD turns 'how to parallelize' into a runtime layout choice, not a code fork."*

支持的并行策略（`components/distributed/`）：
- **FSDP2**（`fsdp2.py`）：HSDP、sequence parallel、activation checkpointing、CPU offload、
  `defer_fsdp_grad_sync`、forward/backward prefetch depth；
- **MegatronFSDP**、**DDP**：三种策略三选一（`DistributedStrategyConfig = Union[FSDP2Config, MegatronFSDPConfig, DDPConfig]`）；
- **TP**：`torch.distributed.tensor.parallel` 的 Colwise/Rowwise/SequenceParallel + 自研 `TPLinear`
  （解决 `F.linear` 在 DTensor + `torch.compile` 下 view 算子 sharding 传播死循环，用 bmm 绕过，
  `parallel_styles.py` 有长篇注释）；
- **PP**：torch-native pipelining（`pipelining/`，AutoPipeline），可与 FSDP2/DTensor 组成 3D 并行；
- **CP**：context parallel（ring attention、`context_parallel/`、`blockdiag_cp/`、THD packed）；
- **EP**：MoE expert parallelism（`moe/`，DeepEP / HybridEP / UCCL-EP）。

mesh 轴名（`mesh.py::MeshAxisName`）：`pp / dp / dp_replicate / dp_shard / dp_shard_cp / dp_cp / cp / tp / ep / ep_shard`。

## 3. 仓库结构：三层解耦

```
automodel <config.yaml> [--nproc-per-node N]
    |
    v
cli/app.py            -- 解析 config 的 recipe target，dispatch launcher
    |
    v
recipes/              -- 端到端 workflow（llm/vlm/diffusion/dllm/multimodal/retrieval）
    |
    v
components/           -- 18 个自包含模块，无跨模块导入
    |
    v
_transformers/ / _diffusers_   -- HF 生态桥接
```

`nemo_automodel` 子包职责：

| 子包 | 职责 |
|---|---|
| `_transformers` | HF Auto-class 兼容入口：`NeMoAutoModelForCausalLM/ForImageTextToText/ForDiffusion/...`、`NeMoAutoTokenizer`、`AutoMFU`，与 `transformers` 同签名 |
| `_diffusers` | `NeMoAutoDiffusionPipeline`（diffusers 管道封装） |
| `components` | 18 个模块：`_peft`（LoRA）、`attention`、`checkpoint`、`config`、`datasets`、`distributed`、`eval`、`flow_matching`、`launcher`、`loggers`、`loss`、`models`、`moe`、`optim`、`quantization`、`speculative`、`training`、`utils`；单元测试与组件 colocated |
| `recipes` | 端到端 workflow：`llm/`（train_ft、kd、benchmark、EAGLE/DFlash/DSpark 系列 speculative 训练…）、`vlm/`、`diffusion/`、`dllm/`、`multimodal/`、`retrieval/` |
| `shared` | 跨模块小工具与补丁（`tp_linear`、`transformers_patches`、`te_patches`、`tied_weights`…） |
| `cli` | `automodel`/`am` 入口 → `nemo_automodel.cli.app:main`，dispatch 本地交互 / SkyPilot / NeMo-Run launcher |
| `autonvtx` | 自动 NVTX range hook（profiling 用） |

设计上 recipes 是**线性 Python 脚本而非 Trainer 类**——训练循环永远可见、可 hack。

## 4. 用户界面：YAML-driven recipes

"Minimal ceremony"：YAML 声明式配置 + CLI override。跑一个 finetune：

```bash
automodel --nproc-per-node 2 examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
```

以 `llama3_2_1b_squad.yaml` 为例，yaml 里全是声明式配置：`recipe: TrainFinetuneRecipeForNextTokenPrediction`、
`step_scheduler`（global/local batch、ckpt/val 频率）、`model._target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained`
（hydra 风格 `_target_`）、`distributed: {strategy: fsdp2, tp_size: 1, cp_size: 1}`、`dataset._target_`、
`optimizer`、`lr_scheduler`、`compile`、`packed_sequence`、`clip_grad_norm`。根目录 `app.py` 是源码 checkout 的便捷入口
（`python app.py <config.yaml>` 等价于 `automodel`）。

## 5. 模型覆盖：三线并进

`docs/model-coverage/` 按模态分 `llm/`、`vlm/`、`diffusion/`、`dllm/`、`omni/`、`embedding/`、`reranker/`、`multimodal/`，
下再按 **org 分目录**（如 `llm/deepseek-ai/`），每个模型一篇 mdx（含 recipe 链接、支持的并行配置、实测环境）。

2026 年中至今的新增重心是 **MoE + VLM + diffusion**：
- MoE：Kimi K3（2.8T，256×GB200）、Qwen3.8-2.4T-A95B、GLM-5.3、MiMo-V2.5-Pro、DeepSeek-V4.1-Flash，普遍配 EP64 + PP/CP；
- VLM：Nemotron-3-Nano-Omni（三模态）、Qwen3.8-27B、Inkling、MuseGlimmer；
- Diffusion：HunyuanImage-3.0（80B-A13B MoE）、Qwen-Image-2.1、DiffusionGemma；另有 dLLM（LLaDA）。
- 共性：hybrid attention、FP8、MTP、超长上下文。

## 6. 特色功能

- **Speculative decoding 训练全家桶**（`examples/speculative/README.md`）：EAGLE-1/2/3/3.1、P-EAGLE、
  **DFlash 2**（two-tap dynamic conv + pairwise path selector）、Domino、JetSpec、**DSpark**
  （DeepSeek 新发布的 semi-autoregressive drafting）、ViSpec；训完的 drafter 去 SGLang/vLLM 里和 target 一起 serve；
- **MTP**（multi-token prediction）：`train_ft.py` 等 recipe 支持；
- **LoRA/PEFT**（`components/_peft/`）：含 MoE 专家 LoRA、MXFP4、VLM QLoRA，每个模型基本有配套 `_lora.yaml`；
- **Packed sequences**：THD packed（`distributed/thd_utils.py`）、packed CP；最新 commit（`7227ea6`）
  正是 "accelerate packed Lightning SFT"。

## 7. skills/：把开发流程做成 agent 可调用的

5 个 public skill（可同步到 Claude Code 公共 catalog，`/<skill-name>` 调用）+ `.agents/contributor-skills/` 6 个内部 skill：

| Skill | 作用 |
|---|---|
| `nemo-automodel-model-onboarding` | 新模型架构接入：五阶段流程 + `llm/moe/vlm-patterns.md` 模式文档 + `BENCHMARK.md` |
| `nemo-automodel-recipe-development` | 创建/修改训练与评测 recipe |
| `nemo-automodel-distributed-training` | FSDP2、HSDP、pipeline/context 并行 |
| `nemo-automodel-launcher-config` | Slurm 与 SkyPilot 任务提交 |
| `pr-review` | 正式 `/review` 的 rubric：light/strict 两档；硬性规则包括读全量 changed-file（`uv.lock` 完全忽略）、数值正确性核查、测试必须断言外部可观察行为 |

`pr-review` 的**防投毒设计**值得注意：从 protected default branch 加载、PR 内容视为不可信输入
（`disable-model-invocation: true`，`user_invocable: false`）。

## 8. 给外部贡献者的规则（`CONTRIBUTING.md` + `AGENTS.md`）

- **DCO sign-off 强制**：无 sign-off 的 commit 不收（`git commit -s`）；未提 CLA；Apache-2.0；
- 环境：Automodel container / `uv sync` / 自定义 docker 三选一；
- 代码规范：ruff（line length 120、double quotes）、每个 Python 文件 NVIDIA copyright header、
  public API 必须 type hints、Google 风格 docstring；
- 测试分层：`tests/unit_tests`（CPU）/ `functional_tests`（GPU，CI 上限 2 卡）/ `ci_tests`；
  tier 语义 L0（每 PR 必过）/L1/L2；
- 分支名 `<github-handle>/<type>/<short-desc>`，commit/PR 标题 Conventional Commits（CI 校验）；
- `AGENTS.md` 强制工作流：动手前先读相关 skill（testing/linting/pr-review），缺一不可。

## 9. 与 Molt / OpenRLHF 的关系（诚实结论）

- **Molt**：本仓库内**零引用**（`grep -ril molt` 无结果）。"AutoModel 是 Molt 训练后端"的说法在当前 repo
  里找不到依据——如需确认，需另查 Molt 侧文档；
- **OpenRLHF / TRL**：仅作为**消费方**出现。`docs/index.mdx`："Drop-in accelerated backend for TRL,
  lm-eval-harness, OpenRLHF, or any code that loads Hugging Face models."——因为 AutoModel 实现了 HF
  `AutoModel` API，RL 框架可直接把 policy/reference 模型换成它来加速，**AutoModel 本体不做 RL**
  （无 GRPO/DPO/PPO 训练循环；`tutorials/README.md` 把 RLHF/DPO/PPO 明确指向 NeMo-RL）。

## 10. 三个设计亮点与两个局限

**亮点**：
1. **HF API 兼容是架构决策而非口号**：`NeMoAutoModelForCausalLM.from_pretrained` 与 `transformers`
   同签名，优化实现（fused attention、TE、DeepEP、FlexAttn）对调用方透明——这正是它能当 TRL/OpenRLHF
   drop-in 后端的原因；
2. **SPMD mesh + 无交叉导入的 components + 线性 recipe 脚本**：三层解耦让"换并行策略"和"换 loss"
   都是配置级操作，研究原型成本极低；
3. **把 speculative decoding 做成一等训练栈 + skills 即开发流程**：从 EAGLE 到 DSpark 的完整
   drafter 训练/评测/serve 对接。

**局限**：
1. **深度绑定 NVIDIA 软硬件新特性**：DeepEP/HybridEP、MXFP8 MoE、GB200 等 recipe 依赖特定硬件与软件栈，
   非 NVIDIA / 老卡用户很多 recipe 跑不起来，大规模验证门槛高；
2. **"Day-0 能用"≠"Day-0 最优"，且 API 快速变化**：新架构先保 correctness、优化内核后补；
   `docs/breaking-changes.mdx` 的存在说明 breaking 变更并不罕见，main 迭代极快，跟进 upstream 需要持续投入。

---

*文件索引：`README.md`、`docs/about/index.mdx`、`docs/index.mdx`、`docs/repository-structure.mdx`、
`nemo_automodel/components/distributed/{mesh,config,fsdp2,megatron_fsdp}.py`、
`nemo_automodel/components/_peft/`、`nemo_automodel/cli/app.py`、`app.py`、
`examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml`、`examples/speculative/README.md`、
`skills/README.md`、`skills/pr-review/SKILL.md`、`CONTRIBUTING.md`、`AGENTS.md`*
