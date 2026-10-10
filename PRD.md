# Snake-RL：从 Laya 推理基线到 BattleSnake 多智能体强化学习

**产品需求文档（PRD）**  

| 项目字段 | 内容                                                 |
| -------- | ---------------------------------------------------- |
| 文档版本 | v0.1.0                                               |
| 状态     | Draft：可用于启动工程预研与阶段零验收                |
| 日期     | 2026-10-10                                           |
| 项目类型 | 个人研究项目 / 强化学习实验系统                      |
| 主要平台 | Windows 11 + WSL2 Ubuntu；NVIDIA RTX 4060 Laptop 8GB |
| 核心技术 | PyTorch、PPO、IPPO、多智能体环境、规则基线、离线评估 |

## 1. 项目概述

### 1.1 一句话定义

构建一条可复现的 Snake / BattleSnake 强化学习实验链路：先跑通 Laya 的 CUDA 推理演示并理解规划器与模型的分工，再在统一的游戏环境中实现 PPO 与 IPPO，最后通过规则策略、随机合法策略、冻结模型和在线对局检验学到的策略。

### 1.2 项目背景

Laya 的贪吃蛇演示将确定性路径规划、结构化决策模型和安全执行层组合在一起。Laya 本身不生成解释文本，也不独立完成全部路径规划；它根据规划器提供的候选方向描述输出结构化概率。该演示适合用来研究推理和安全控制边界，但不是 RL 训练系统。

本项目进一步提出一个研究问题：**在同一个可控环境中，规则规划、Laya 决策模型与通过 PPO / IPPO 训练的策略各自能做到什么？** 要回答这个问题，需要统一环境规则、观测编码、动作空间、奖励定义和评估协议，避免把“环境不同”或“奖励写法不同”误判成模型能力提升。

### 1.3 北极星目标

交付一套端到端、可复现实验系统，使每个结果都能追溯到环境版本、策略版本、训练配置、随机种子和评估对手，并能回答：

1. PPO 是否超过随机合法动作基线？
2. 多智能体策略是否超过规则基线和固定对手？
3. 提升来自策略学习，还是来自奖励塑形、环境差异或安全过滤？
4. 推理延迟、环境吞吐量和评估稳定性是否支持后续在线部署？

## 2. 目标与非目标

### 2.1 项目目标

- **G1：验证推理环境。** 在 RTX 4060 上运行 `laya-snake-cuda`，确认 CUDA、模型权重下载、推理与无界面测试可用。
- **G2：建立可复现环境接口。** 固定一个主环境库，验证 reset / observe / action mask / step / reward / terminal 的真实语义。
- **G3：跑通单智能体 PPO。** 完成 rollout、优势估计、策略更新、价值函数更新、checkpoint 保存和独立评估。
- **G4：训练竞争策略。** 从 1v1 固定对手开始，再逐步扩展到四蛇 Standard 场景，优先实现 IPPO。
- **G5：建立可解释的对照实验。** 至少比较随机合法策略、规则基线、Laya 辅助决策基线和 RL 策略。
- **G6：让结论可复核。** 保存配置、种子、原始对局、指标和策略版本，不用训练 loss 单独宣称成功。

### 2.2 非目标

- 不从头训练 Laya，也不在本项目第一阶段复现 RLCD。
- 不把 `laya-snake-cuda` 当成 PPO / IPPO 训练框架。
- 不在阶段零之前同时引入多个模拟器、多个训练框架和大规模并行。
- 不把哈密顿回路上的 `advance` 最大化直接视作完整的多蛇对抗策略。
- 不以单次高分、单个随机种子或训练 loss 下降证明策略学会了泛化博弈。
- 不在离线评估尚未稳定之前把在线竞赛部署作为主开发任务。

## 3. 用户与关键使用场景

### 3.1 主要用户

项目开发者本人，目标是通过构建真实的环境—采样—奖励—训练—评估闭环，掌握 RL 系统工程，并研究从规则决策到多智能体博弈的策略差异。

### 3.2 用户故事

- 作为研究者，我希望先用最小测试确认环境 API、动作掩码与奖励语义，避免在错误接口上投入训练时间。
- 作为实验操作者，我希望训练运行可以固定种子并保存完整配置，以复现某次模型表现。
- 作为评估者，我希望在同一批初始条件和对手上比较不同策略，而不是只看训练过程的回报曲线。
- 作为系统维护者，我希望可以替换环境适配器或策略模块，而无需重写整个训练与评估流程。
- 作为在线部署开发者，我希望策略的动作空间与 Battlesnake API 一致，并有清楚的超时、非法动作和异常处理。

## 4. 设计原则

1. **先正确，再加速。** 先用 FP32 和少量环境跑通训练数学与数据流，再评估 BF16、编译和并行采样。
2. **Environment 与 Verifier 分离。** 环境推进真实状态；独立验证器检查终止、碰撞、动作映射、回报与不变量。
3. **规则策略与 Laya 模型分开记账。** 纯规则规划器和使用 Laya 选择动作的策略必须分别命名、分别评估。
4. **奖励定义显式化。** 原环境内置奖励、稀疏奖励和额外塑形奖励用不同版本号，不混称为同一实验。
5. **动作掩码语义必须精确。** “环境认为动作未被判定为必死”不等于“动作绝对安全”。
6. **每个结论都需要对照。** 至少包含随机合法策略与规则基线；多智能体实验必须指定对手池。
7. **单一主环境。** 训练阶段不要混用不同模拟器的 rollout；如果更换环境，必须从新环境重新训练或做明确的跨环境迁移实验。

## 5. 技术决策与环境选型

### 5.1 主环境候选

当前首选候选为 [PyPZBattleSnake](https://pypi.org/project/pypzbattlesnake/)，原因是其公开文档包含 `BattleSnake` 环境、组队配置、离散观测、动作掩码以及食物接近奖励。它适合快速验证 PPO / IPPO 所需的关键接口。

但在阶段零通过前，**该库仅是候选，不视为已确认的最终依赖**。PyPI 当前公开版本较旧，必须实际验证当前 Python / PyTorch 环境能否稳定使用。

### 5.2 必须澄清的 API 事实

- `pypzbattlesnake` 的文档展示的是 `from pypzbattlesnake import BattleSnake`；不能误写成 `from pypzbattlesnake.env import solo_v0`。
- `solo_v0` / `standard_v0` 风格的导入接口属于另一个项目 [DaBultz/pz-battlesnake](https://dabultz.github.io/pz-battlesnake/environments/standard/)。如果选择该项目，Solo、Duels、Standard 必须全部使用它的接口，不得把两个包混装成一个环境抽象。
- PyPZBattleSnake 公开文档列出五个动作：UP、DOWN、LEFT、RIGHT、NONE；在线 Battlesnake 服务只接受四个方向。因此在线兼容模式必须明确屏蔽或映射 `NONE`，不能将它静默映射为任一方向。
- `action_masks()` 返回的 `False` 表示环境认为该动作会导致失败；`True` 不保证存活。它不是完整的“安全证明”。
- `food_reward` 包含吃到食物的奖励，也包含靠近食物的逐步奖励，以及远离食物的对应惩罚。使用默认值的实验必须标记为“环境内置塑形奖励”，不能称为纯稀疏奖励。

### 5.3 阶段零的环境锁定规则

阶段零需要验证：

1. 能创建 1 蛇、2 蛇和 4 蛇配置；若 1 蛇配置不支持，应明确记录并选用同一项目中的 Solo 接口或切换主环境。
2. 能完整运行一局，返回观测、奖励、终止状态和动作掩码。
3. 观测空间、动作顺序、动作含义、蛇头/蛇身编码和食物编码均有测试，不依赖猜测。
4. 固定种子时，环境的可复现程度经过实测；若无法完全确定性复现，记录随机源及偏差。
5. 记录每步的 CPU 时间、环境吞吐量、进程内存和异常率。
6. 确认多蛇动作在同一状态下如何提交，以及环境是否按同步回合推进。

若候选环境不能通过上述测试，应在开始训练前切换到 [DaBultz/pz-battlesnake](https://github.com/DaBultz/pz-battlesnake) 或评估 [Hisss](https://github.com/ymahlau/hisss)，并在整个训练—评估主链路中统一使用最终选定的环境。Hisss 是高性能 C++ BattleSnake 模拟器，可以作为后续吞吐量对照，但初版不同时维护多套环境。

### 5.4 Laya 推理候选

使用 [inhabitants/laya-snake-cuda](https://github.com/inhabitants/laya-snake-cuda) 进行 NVIDIA / CPU 推理验证。该仓库 README 当前标记为“已发布但不再维护”，因此它是学习与基线工具，不作为必须长期依赖的训练框架。

当前 README 的安装方式为：安装适配当前 GPU 的 PyTorch CUDA wheel，随后 `pip install -e .`、`laya-snake download`、`laya-snake`。RTX 4060 的实际可用性必须通过本机 `torch.cuda.is_available()`、模型加载和 headless 测试确认，不能直接复制其他 GPU 的性能数据。

## 6. 系统架构

### 6.1 数据流

```text
Game Environment
  └─ state / observation / legal-action mask / reward / done
       ↓
Environment Adapter + Versioned Observation Encoder
       ↓
Actor-Critic Policy (masked action distribution)
       ↓
Action Adapter → Environment Step
       ↓
Rollout Buffer → PPO / IPPO Update
       ↓
Checkpoint + Evaluation Runner
       ↓
Independent Verifier + Experiment Ledger + Replay
```

旁路基线：

- `RandomLegalBaseline`：从允许动作中随机选择。
- `RuleBaseline`：使用合法动作、食物路径、可达空间和碰撞风险的确定性规则。
- `LayaDecisionBaseline`：把候选动作与结构化文字描述交给 Laya，再选择动作；与纯规则策略分别评估。
- `TrainedPolicy`：PPO / IPPO checkpoint。

### 6.2 模块职责

| 模块                | 职责                                                     | 不负责什么                              |
| ------------------- | -------------------------------------------------------- | --------------------------------------- |
| Environment Adapter | 隔离环境库 API；统一 reset、observe、step、reward、done  | 不重写游戏规则                          |
| Observation Encoder | 将原始离散棋盘映射到版本化张量 / 特征平面                | 不凭空添加环境未提供的信息              |
| Action Mask Adapter | 对动作掩码顺序、合法性和在线动作集合进行统一处理         | 不声称所有未屏蔽动作都安全              |
| Actor-Critic        | 输出动作分布与价值估计                                   | 不负责推进环境                          |
| PPO Trainer         | rollout、GAE、损失计算、更新、保存 checkpoint            | 不以训练 loss 作为最终成功标准          |
| IPPO Trainer        | 每个智能体依据自身观测与奖励更新策略；保存对手与策略版本 | 不假设对手分布平稳                      |
| Baseline Suite      | 随机、规则、Laya 决策和历史 checkpoint 对照              | 不与待测模型共享未声明的信息            |
| Verifier            | 校验动作、回报、终止、记录完整性及评估协议               | 不使用待测策略自报结果作为 ground truth |
| Experiment Ledger   | 保存配置、版本、随机种子、训练/评估指标和产物路径        | 不只保存汇总分数                        |

## 7. 功能需求

### FR-01：环境适配层

- 提供统一的 `reset(seed)`、`observe(agent_id)`、`action_mask(agent_id)`、`step(actions)`、`episode_end()` 语义。
- 适配器必须显式记录原环境 API 的动作顺序和奖励结构。
- 单蛇、1v1 和四蛇场景使用同一个主环境实现；环境迁移需要提高文档版本并重跑全部基线。

### FR-02：版本化观测编码

- 第一次接入时先检查原始 observation 的 shape、dtype、取值范围和类别含义。
- 将离散棋盘映射为确定的特征平面或嵌入，记录 `observation_schema_version`。
- 不在 API 尚未验证前固定“五通道 20×20”设计。初始 9×9 环境应保留原尺寸或使用明确记录的 padding / resize 规则。
- 如进行旋转或镜像增强，动作标签、动作掩码和观测必须同步变换并通过单元测试。

### FR-03：动作空间与掩码

- PPO 的 `log_prob`、熵和采样必须采用一致的 masked distribution。
- 掩码语义、动作索引和环境动作之间必须有可测试的双向映射。
- 训练模式如包含 `NONE`，须明确其用途；在线兼容模式只输出 up / down / left / right。
- 若掩码过滤后没有可采样的方向，必须进入显式 fallback 分支并记录 `forced_loss_or_fallback`，不得向全为 `-inf` 的 logits 计算 softmax。
- 每局记录非法动作次数，目标为 0。

### FR-04：单智能体 PPO

- 实现 rollout 收集、折扣回报、GAE、clipped policy objective、value loss、entropy bonus、梯度裁剪和 checkpoint 保存。
- actor 与 critic 可以共享 CNN 特征骨干，但实现必须显式记录网络结构和参数量。
- 第一版使用 FP32；只有在训练数值、吞吐量和显存测量通过后，才引入 BF16 / AMP。
- 训练和评估环境、随机种子池必须分离。

### FR-05：多智能体 IPPO

- 第一目标是 1v1；通过后再扩展到四蛇 Standard。
- 每个智能体使用独立的 actor / critic 参数作为基础 IPPO 方案；参数共享可作为后续独立实验，不与 IPPO 主结果混标。
- 训练时显式记录各智能体观测、动作、动作掩码、奖励、done、价值估计和对手策略版本。
- 支持固定规则对手、随机合法对手、历史 checkpoint 对手，以及阶段性 self-play 对手池。
- 初版暂不实现 MAPPO centralized critic；只有 IPPO 基线已经可靠后再评估。

### FR-06：奖励配置版本化

至少支持以下独立配置：

- `R0_native`：保留环境原生奖励，必须记录其中是否包含靠近 / 远离食物的塑形。
- `R1_terminal_outcome`：以胜 / 平 / 负为主的结果奖励，作为竞争目标的对照。
- `R2_shaping_ablation`：在主奖励上增加有界的距离或可达性塑形；单独标版本并与无塑形实验比较。
- `R3_kill_bonus_experiment`：击杀相关奖励仅作为独立消融，不是默认奖励。

对比奖励时固定环境、初始种子、对手池、网络结构和训练预算。不得只因密集奖励的 episode return 更高，就宣称策略更强。

### FR-07：基线策略

- `RandomLegalBaseline`：在动作掩码允许的方向中随机选择。
- `RuleBaseline`：使用当前多蛇状态实现食物路径、安全区域和碰撞风险评估；不得把原始单蛇哈密顿回路简单复制为多蛇策略。
- `LayaDecisionBaseline`：独立适配候选动作与文字提示，调用冻结的 Laya 推理；记录模型版本、提示模板和延迟。
- 所有基线均应与 RL 策略在相同游戏规则和相同评估协议下比较。

### FR-08：评估器与验证器

- Solo 指标：食物数量、存活步数、终止原因、非法动作数、单位食物所需步数。
- 多智能体指标：胜率、平均名次、存活时长、吃到食物数量、头对头碰撞、撞墙 / 撞身死亡、非法动作数。
- 记录均值、中位数、分位数和置信区间；多随机种子运行，防止只报告最好的一次。
- 评估代码必须独立于训练循环；每次发布结果保存原始对局或可重放记录。

### FR-09：实验账本与可复现性

每次运行至少记录：`run_id`、git commit、Python / PyTorch / CUDA / 环境包版本、配置文件、训练种子、评估种子、地图尺寸、环境模式、观测 schema、奖励版本、策略 checkpoint hash、对手版本、训练步数、环境吞吐量、显存峰值、系统内存峰值、结果文件路径。

数据结构建议使用 JSONL 记录逐局结果，YAML 保存运行配置；TensorBoard 可作为本地曲线展示，不作为唯一记录源。

### FR-10：在线 API 封装（后置）

只有离线评估通过后才实现在线服务。服务层负责解析 Battlesnake 请求、构造观测、加载模型、在时限内返回四方向动作，以及处理异常、超时和非法动作。线上适配层必须与训练动作空间保持一致。

## 8. 分阶段交付与验收标准

### 阶段零：环境接口与资源预检

**目的：** 在任何长时间训练之前，先证明环境真实可用。

**任务：**

1. 安装并测试候选环境；核对 1 蛇、2 蛇、4 蛇配置。
2. 逐字段记录 observation schema、动作索引与 mask 顺序。
3. 运行完整 episode，检查 reward / done / terminal reason。
4. 测量 environment steps / second、CPU 时间、RAM 和异常率。
5. 验证固定随机种子的重复性和同步动作语义。

**验收：** 接口测试全部通过；无未解释的动作索引；每局可生成完整日志；奖励定义有文档；主环境和包版本已锁定。未通过则不得进入训练阶段。

### 阶段一：Laya CUDA 推理基线

**目的：** 确认 RTX 4060 能运行冻结的 Laya 模型，并理解原演示中的规划器、模型和安全层的分工。

**任务：**

1. 按 `laya-snake-cuda` 当前 README 安装适配当前显卡的 PyTorch CUDA 版本。
2. 下载并加载模型，在 GPU 上运行交互模式和 headless 模式。
3. 保存至少一段 JSONL 运行记录；验证推理概率、动作和安全层干预字段。
4. 分别观察 shield 开启与 `--unassisted` 的动作差异；可使用扰动模式理解固定路线与恢复规划器。
5. 记录本机实测延迟、VRAM 和吞吐量，不套用其他 GPU 的公开数字。

**验收：** CUDA 确认为可用；headless 连续运行至少 100 步不崩溃；模型可重复加载；运行记录可导出或检查；清楚标注它是冻结推理基线，不是 RL 训练结果。

### 阶段二：单智能体 PPO

**目的：** 跑通完整 on-policy RL 循环，并验证策略是否优于随机合法策略。

**初始设定：** 9×9 棋盘（如环境支持）、小型 CNN actor-critic、FP32、少量并行环境；初始先从 4 个环境开始，不默认使用 8–16 个 subprocess。

**任务：**

1. 完成观测编码和动作 mask 单元测试。
2. 实现 rollout buffer、GAE、PPO 更新和 checkpoint。
3. 先用 `R0_native` 跑通，再运行单独的稀疏 / 塑形奖励对照。
4. 使用固定评估种子与随机合法策略比较。
5. 记录样本吞吐量、显存和系统内存，再决定是否并行化与 AMP。

**验收：** 至少 3 个独立训练种子；每个 checkpoint 使用固定且不参与训练的评估种子池；PPO 在食物数量或存活指标上稳定优于随机合法基线，且差异不是仅由单个种子驱动；训练结果、配置和日志可以重新加载。若未达标，优先检查奖励、动作掩码、GAE 和 done 处理，而不是直接扩大训练规模。

### 阶段三：IPPO 多智能体训练

**目的：** 从单蛇生存转向竞争结果，验证策略在非固定对手下的表现。

**顺序：**

1. 1v1 对固定随机 / 规则对手。
2. 引入冻结的历史 checkpoint 对手。
3. 开启受控 self-play 对手池。
4. 再扩展到四蛇 Standard。
5. 最后对奖励塑形做消融；MAPPO 暂不属于 MVP。

**验收：** 至少对随机合法策略、规则基线和历史策略进行评估；每种主要对手至少 300 局或给出受限预算的明确说明；报告胜率 / 名次的置信区间与死亡原因分布；训练与评估对手严格分离；没有动作掩码或回合推进语义不一致的问题。

### 阶段四：基线对照与在线验证

**目的：** 判断学习策略是否具有泛化能力，并确认在线 API 能执行相同动作语义。

**任务：**

1. 固定环境规则、种子、对手池和算力预算，对所有策略进行配对评估。
2. 与随机合法、规则、Laya 决策和历史 checkpoint 对照。
3. 评估旋转 / 镜像变化、不同棋盘尺寸和不同对手强度下的泛化。
4. 离线指标通过后，再部署到 Battlesnake HTTP API，做 smoke test 和少量对局验证。

**验收：** 至少有一组可复核的离线结果显示 RL 在预先指定的竞争指标上优于基线，且结果在多个种子下稳定；在线服务通过请求 / 响应、超时和动作合法性测试。若没有稳定提升，结论应为“当前训练方案未证明优于基线”，不得通过挑选种子得出成功结论。

## 9. 算法与实验设计

### 9.1 PPO 主线

- 使用离散动作 Actor-Critic；策略输出经过 mask 的 categorical distribution。
- 对终止状态、时间上限截断与环境死亡分别记录；正确区分 `terminated` 与 `truncated`，防止错误 bootstrap。
- 首版建议小型 CNN + 128 / 256 维隐藏层，避免一开始固定约 660 万参数的网络。
- 先确认梯度、log probability、advantage 标准化和 checkpoint 恢复正确，再调整 batch size、rollout 长度、epoch 数与学习率。

### 9.2 IPPO 主线

- 第一阶段 1v1，每个 agent 独立 actor / critic。
- 每一轮应根据同一游戏状态收集各蛇动作；环境适配层需验证真实的同步决策语义。
- 对手池包括随机合法、规则策略、Laya 决策策略以及历史 checkpoint；定期在冻结评估对手上报告成绩。
- 如果后续使用参数共享、循环网络或 centralized critic，必须作为新配置单独记录，不能覆盖 IPPO 基线结果。

### 9.3 奖励设计

多蛇竞争的主目标应与胜负 / 名次一致。食物数量和存活时间用于诊断，不一定与获胜率单调一致。

默认环境奖励可能已经含有距离食物的密集塑形。必须先记录实际奖励组成，再引入自定义奖励。距离奖励有可能鼓励蛇靠近食物但忽略路径风险；击杀奖励有可能让策略过度冒险。任何奖励变化都应配套消融实验，并保留终局胜负指标作为独立 ground truth。

## 10. 硬件与性能预算

目标设备为 RTX 4060 Laptop 8GB VRAM。此规模足以尝试小型 CNN 策略，但训练吞吐量也可能受到 Python 环境仿真和系统内存限制。

- 初始使用 FP32；先测量模型与环境的独立耗时，再评估 BF16 / AMP。
- 环境并行从 4 个开始；只有在环境 steps / second 明显提升且 RAM 稳定时再增加到 8 个或更多。
- 不把 8–16 个 `SubprocVecEnv`、大 rollout 和大型网络视为默认配置。
- 每个训练配置都记录 GPU 显存峰值、主进程和子进程 RSS、CPU 利用率与 environment steps / second。
- 若环境成为瓶颈，再单独评估 [Hisss](https://github.com/ymahlau/hisss) 这类 C++ 高吞吐模拟器；环境替换必须重跑基线并标记环境版本变化。

## 11. 验收指标总表

| 维度       | 必须记录的指标                                               | 通过标准                                           |
| ---------- | ------------------------------------------------------------ | -------------------------------------------------- |
| 环境正确性 | reset / step / mask / reward / done 测试、终止原因、动作映射 | 所有接口测试通过，无未解释的非法动作               |
| 推理可用性 | CUDA 可用性、加载成功率、延迟、VRAM、headless 步数           | 连续 100 步无异常，记录可复核                      |
| PPO 学习   | 平均食物数、存活步数、回报、随机基线差值、跨种子差异         | 多种子下稳定优于随机合法基线；仅 loss 下降不算通过 |
| IPPO 竞争  | 胜率、平均名次、存活时长、食物数、死亡原因、置信区间         | 在预先指定的对手池中取得可复核提升                 |
| 可靠性     | 非法动作数、崩溃率、日志完整率、checkpoint 恢复              | 非法动作目标为 0；日志与 checkpoint 能检查和加载   |
| 性能       | 环境 steps / second、回合数 / second、VRAM / RSS、P50 / P95 延迟 | 满足当前实验预算；优化必须有前后对照               |
| 泛化       | 不同种子、对手、棋盘尺寸或对称变换结果                       | 明确报告退化范围，不只报告最佳场景                 |

## 12. 建议代码结构

```text
snake-rl/
├── README.md
├── PRD.md
├── pyproject.toml
├── configs/
│   ├── env_solo_9x9.yaml
│   ├── ppo_baseline.yaml
│   ├── ippo_duel.yaml
│   └── eval.yaml
├── src/snake_rl/
│   ├── envs/adapter.py
│   ├── envs/reward_profiles.py
│   ├── encoders/grid_encoder.py
│   ├── policies/cnn_actor_critic.py
│   ├── trainers/ppo.py
│   ├── trainers/ippo.py
│   ├── baselines/random_legal.py
│   ├── baselines/rule_planner.py
│   ├── baselines/laya_adapter.py
│   ├── evaluation/evaluate.py
│   ├── evaluation/verifier.py
│   └── tracking/ledger.py
├── tests/
│   ├── test_env_contract.py
│   ├── test_action_masks.py
│   ├── test_observation_encoder.py
│   ├── test_reward_profiles.py
│   └── test_reproducibility.py
└── runs/  # 本地生成，不提交大体积产物
    └── <run_id>/
        ├── config.yaml
        ├── train.jsonl
        ├── eval.jsonl
        ├── checkpoints/
        └── replays/
```

## 13. 风险清单

| 风险                            | 影响                       | 缓解方案                                                     |
| ------------------------------- | -------------------------- | ------------------------------------------------------------ |
| 环境包版本旧或 API 与示例不一致 | 训练无法启动或语义错误     | 阶段零先做 API 契约测试，锁包版本和 commit                   |
| `NONE` 与在线四方向不一致       | 训练策略无法直接部署       | 在线兼容模式只输出四方向；显式测试 fallback                  |
| 原生奖励包含距离塑形            | 错判奖励实验结果           | 记录奖励分解；用版本化 reward profile 做消融                 |
| 动作 mask 被误解为绝对安全      | 训练数据和评估出现隐藏失败 | 独立校验动作映射；记录被屏蔽动作与死亡原因                   |
| 对手固定导致过拟合              | 自博弈结果无法泛化         | 使用规则、随机和历史 checkpoint 对手池                       |
| 只看最好种子或训练回报          | 结论不可复现               | 固定评估集、多训练种子、报告置信区间                         |
| 环境仿真慢 / 系统内存不足       | GPU 利用率低或进程崩溃     | 逐步增加并行度；测量 RSS；必要时评估 C++ 模拟器              |
| Laya 推理基线与 RL 输入不同     | 对照不公平                 | 分别定义 RuleBaseline 与 LayaDecisionBaseline，公开输入和输出协议 |
| 直接将哈密顿回路迁移到多蛇竞争  | 规划路线被堵或发生对抗碰撞 | 重建多蛇规则规划基线，不宣称原单蛇安全保证仍成立             |

## 14. 关键决策记录（ADR 摘要）

- **D-01：先做环境预检再训练。** 任何模型训练都以环境接口契约测试通过为前置条件。
- **D-02：PyPZBattleSnake 是候选，不是无条件锁定。** 阶段零通过后锁定版本；失败则选另一套环境并在全链路中统一使用。
- **D-03：Laya 是冻结推理基线。** 不把它当 PPO / IPPO 训练器；Laya 的规则规划器与模型决策分开评估。
- **D-04：先 PPO、后 IPPO。** 先验证单智能体训练循环，再进入 1v1 和四蛇竞争。
- **D-05：MAPPO、RLCD、HILL 和大型自博弈先延后。** 它们不是 MVP 跑通训练—评估闭环的必要条件。
- **D-06：性能优化按测量触发。** 不预设 8–16 环境、BF16 或大型网络一定更好。

## 15. 阶段零任务清单

开始写 PPO 代码前，必须完成以下勾选项：

- [ ] 记录操作系统、Python、PyTorch、CUDA、显卡驱动及环境包版本。
- [ ] 验证 `torch.cuda.is_available()` 和一次 CUDA 张量计算。
- [ ] 运行 Laya CUDA 推理 smoke test；保存输出与实测延迟。
- [ ] 确认主环境的导入路径、安装方式和准确版本。
- [ ] 确认 1 蛇、1v1、四蛇配置及其观测 shape。
- [ ] 验证动作索引与 `action_masks` 顺序；单测 `NONE` 的处理。
- [ ] 记录原生奖励组成、健康值或回合终止逻辑、食物生成逻辑。
- [ ] 运行完整回合并保存逐步 JSONL 记录。
- [ ] 测量环境吞吐量、RAM / RSS、VRAM 和失败率。
- [ ] 通过审阅后锁定环境库与版本；不得同时使用两种环境 API。

## 16. 参考资料

以下仓库与文档用于技术选型。依赖版本、安装命令与维护状态应在实现前再次核对。

1. [Laya-MLX — 上游结构化决策与 Snake Demo](https://github.com/mizorewww/laya-mlx)
2. [laya-snake-cuda — PyTorch / NVIDIA 移植版及 README](https://github.com/inhabitants/laya-snake-cuda)（README 当前标注为不再维护）
3. [PyPZBattleSnake — PyPI 项目文档](https://pypi.org/project/pypzbattlesnake/)（包含奖励、动作空间与 mask 说明）
4. [DaBultz/pz-battlesnake — PettingZoo 风格环境文档](https://dabultz.github.io/pz-battlesnake/environments/standard/)
5. [Hisss — High-Speed Snake Simulator](https://github.com/ymahlau/hisss)（可选的高吞吐 C++ 模拟器）
6. [AWS SageMaker BattleSnake AI](https://github.com/awslabs/sagemaker-battlesnake-ai)（已归档，仅作为历史实现参考）
7. [BattleSnake 官方平台](https://play.battlesnake.com/)

---

**文档维护规则：** 当环境库、奖励定义、观测编码、动作空间或主要训练算法发生改变时，更新 PRD 版本与决策记录；不得只修改代码而不更新实验协议。
