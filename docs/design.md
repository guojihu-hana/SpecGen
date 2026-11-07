# SpecGen 设计

SpecGen 用推理型 LLM 迭代生成并优化 GPU kernel。主模型推理的同时，根据推理内容实时发起推测生成（speculative generation），把候选 kernel 交给验证和性能测量流水线，并可在达到条件时提前结束本轮迭代。

## 1. 架构

```text
IterationLoop（每个任务一个；所有任务共享资源池、SpecBudget 和 LLM 服务）
  ① PromptBuilder：任务 + 上一轮最快的若干 kernel（计时和 ncu 摘要）+ 历史最优
  ② Generator.stream_main（流式；由 ModelProfile 拆出 reasoning 和正文）
        └─ reasoning 增量 ─► 按需调度（默认）：预计到达 profiler 的 kernel 不足时发 1 个 generate_spec，
              推理前缀后附本轮已完成 spec 的测试结果；或触发调度：TriggerMonitor 触发 ─► SpecBudget 定 k ─► k 个
  ③ 主 kernel 和 spec kernel ─► 去重 ─► Builder（CPU 池）─► Validator 池（每卡多并发）
        └─ 通过 ─► Profiler 池：计时（独占）─► TerminationController
                        └─ 本轮最快的前 N 个 ─► ncu（独占，有空闲才做）
  ④ 结束：满足终止条件立即结束；否则等主 kernel 评测完，其余候选最多再等主生成时长的 grace_ratio 倍

PerfRecorder：记录所有事件，提供耗时、通过率、并发度、利用率的滑动统计
```

- 编排层基于 asyncio：LLM 流式输出、spec 请求、子进程管理和取消都是异步任务，结束一轮迭代就是取消这一组任务。
- LLM 调用在进程内执行，才能拿到实时 token 事件。
- GPU 任务在隔离子进程中运行：固定 `CUDA_VISIBLE_DEVICES`，每个 kernel 独立的 `TORCH_EXTENSIONS_DIR`，带超时，可被杀掉（整个进程组）。

## 2. 组件

### 生成

- **ModelProfile**：每个模型一份配置（`configs/models.yaml`），描述推理内容的位置（`field` 单独字段，或 `inline` 夹在正文的思考标签里）、思考标签、分词器、采样参数，以及自部署和 API 两个端点。地址和模型名用占位符，API key 只通过环境变量传入。
- **生成模式（`modes`）**：每种生成（`main` / `spec`）附加的请求参数，可写在模型层或端点层（端点层优先）。例如 DeepSeek API：`main` 为 `thinking: enabled` + `reasoning_effort: max`，`spec` 为 `thinking: disabled`。主生成被 max_tokens 截断后的续写请求也按 `spec` 模式发送（关闭思考，直接根据已有推理写代码）。
- **ChatBackend**：OpenAI 兼容实现覆盖 vllm、sglang、api（openai、deepseek、fireworks、sambanova 等，只是 base_url 和 key 不同）和 together；Anthropic（Messages API，thinking 块作为推理，任务和历史最优两段设缓存断点）、Gemini（`streamGenerateContent`，`thought` 部分作为推理）、file_queue（请求和响应都是共享目录中的文件，响应可以逐行流式追加）各一个实现，均直接用 httpx。输出统一的事件流：`ReasoningDelta`、`ContentDelta`、`UsageUpdate`、`StreamEnd`。
- **Generator**：主生成只用流式；被 max_tokens 截断时可按配置续写。spec 生成支持两种 prompt 模式：
  - `prefill`：最后一条 assistant 消息为 `<think>\n{推理前缀}\n</think>\n\n```python\n`，模型从代码块开头直接续写；输出前补回预填的代码块开头再提取代码，未闭合的代码块也接受。端点在模型配置里声明续写方式（`prefill.style`）：`deepseek`（DeepSeek 的 `/beta` 地址，最后一条消息带 `prefix: true`，已实测可用）、`vllm`（`continue_final_message`，需先用探测脚本确认模板保留思考内容）、`native`（API 本身就续写末尾的 assistant 消息，如 Anthropic）。
  - `reprompt`：把推理前缀和 spec 指令追加在主 prompt 之后重新发送。
  - `auto`（默认）：端点声明了续写方式就用 prefill，否则 reprompt。
  - 同一次触发的第 j 个 spec 采样温度为 `spec_temperature + j × spec_temperature_step`（上限 `spec_temperature_max`），输出上限 `spec_max_tokens`；spec 请求用 `spec` 模式参数（DeepSeek 关闭 thinking）。
- **TokenCounter**：优先用服务端实时用量（vLLM `stream_options.continuous_usage_stats`）；否则用分词器对增量计数。目前统一使用通用分词器 tiktoken `o200k_base`，不可用时退回按字符数估算。增量计数只在"非空白字符后的空格"处切分，结果与整段计数一致。实测 DeepSeek API 只在最后一个片段上报用量，通用分词器的计数约为服务端的 95%。

### 评测

- **编译架构**：用 GPU 评测时必须指定 `--cuda-arch`：H100/H200 为 `9.0a`，B200 为 `10.0a`，B300 为 `10.3a`；也可以显式写 `auto`，按每张卡检测到的计算能力设置（sm_90 及以上用 `a` 变体）。评测子进程只用这个列表，不继承环境里的多架构列表，否则每次编译都要编多个架构，且 `cp.async`、wgmma 这类架构专属代码会被误判为编译错误。启动时检查列表能否在每张卡上运行（`a` 变体只能用于完全相同的计算能力），例如在 H100 上误用 B300 的 `10.3a` 会直接报错。
- **GPU 型号**：`--gpu-type` 决定 prompt 中的硬件参数表（含 B200、B300 和 A800；A800 的算力、显存带宽同 A100 SXM4 80GB，NVLink 削减到 400 GB/s）。默认按第一张卡检测，型号不在表中时报错并要求显式指定，不再静默当作 H100。
- **编译并行**：每个 kernel 的 `load_inline` 构建只有两个编译单元（`main.cpp`、`cuda.cu`）加一次链接，ninja 单次构建最多并行 2 个任务。真正的提速来自多个 kernel 同时构建，所以要避免每个构建都按全部核数起任务：子进程的 `MAX_JOBS` 取有效 CPU 数（亲和性掩码与 cgroup 配额中的较小值）除以可同时构建的槽位数，限制在 2 到 32 之间，可用 `--build-jobs` 覆盖。
- **Builder**：资源池里的 `builder` 角色（CPU，`--build-workers` 个并发，默认有效 CPU 数 / 2、最多 64，每个构建 2 个编译任务（`main.cpp` 和 `cuda.cu` 并行）；纯 CPU 运行时关闭）。子进程在没有 GPU 的情况下导入 kernel 模块，`load_inline` 用验证卡的架构列表编译，产物放在 kernel 独立的扩展目录，验证时 ninja 判断无需重编，直接加载（实测验证阶段编译时间由约 50 秒降到 0.3 秒）。只有确定的编译失败（构建报错、语法错误）和超时直接判失败，导入时需要 GPU 等其他问题留给验证阶段处理。
- **快速构建**（`--fast-cuda-includes`、`--build-pch`，默认都开）：一次构建是 nvcc 编 `cuda.cu`（关键路径）与 g++ 编 `main.cpp`（pybind 绑定）并行，再链接。时间几乎都花在解析 torch 头文件上：`load_inline` 已在 CUDA 源码前加 `<torch/types.h>`（整个 ATen），生成代码又常再引入 `<torch/extension.h>`（整个 C++ API 和 pybind11），这样写 nvcc 要 44 秒，`main.cpp` 16 秒。评测子进程包装 `load_inline`，同一份代码按四级依次尝试：
  0. `bare`：`cuda_sources` 去掉注释和字符串后不出现任何 torch / ATen / c10 / pybind11 的名字、宏和头文件时（system prompt 要求的写法：`.cu` 里只有 kernel 和接收裸指针、`cudaStream_t` 的启动函数，所有碰张量的包装放在 `cpp_sources`），`no_implicit_headers=True`，`.cu` 前只加 CUDA 与 C++ 标准头文件（`<cuda.h>`、`<cuda_runtime.h>`、`<cuda_fp16.h>`、`<cuda_bf16.h>`、`<cstdint>`、`<cmath>` 等），`cpp_sources` 前补回 `<torch/extension.h>`。B300 主机实测同一组 kernel：nvcc 由 `slim` 的 12.7 秒降到 2.2 秒，整次构建 13.3 秒 → 5.8 秒（`main.cpp` 多了包装和 `CUDAContext.h`，4.8 → 5.2 秒），SASS 逐条相同；编不过（例如漏了这里没给的头文件）就退到 `slim`。
  1. `slim`：`no_implicit_headers=True`，`cuda_sources` 前改加精简前导（`<cuda.h>`、`<cuda_runtime.h>`、`<ATen/core/Tensor.h>`、`empty`/`zeros`/`ones`/`full` 系列工厂函数的 `ATen/ops` 头文件，以及 `torch` 命名空间里的 `using namespace at` 和 `kFloat32` 等 dtype 别名），去掉源码中的 `<torch/types.h>`、`<torch/extension.h>`、`<torch/torch.h>`、`<torch/all.h>`、`<ATen/ATen.h>`，并在 `cpp_sources` 前补回 `<torch/extension.h>`（`no_implicit_headers` 同时去掉了 `main.cpp` 的这一行）。前导是 `<torch/types.h>` 的子集，所以用它能编过的代码，按原样也能编过。
  2. `trimmed`：保留自动加的头文件，只去掉 `cuda_sources` 中的 `<torch/extension.h>`、`<torch/torch.h>`、`<torch/all.h>`。
  3. `original`：原样构建。

  某一级报构建错误就以同一扩展名用下一级重编；成功的一级记在扩展目录的 `specgen_fastbuild.json`，后续进程（验证、计时、ncu）直接沿用，保证复用同一产物。代码自己传了 `no_implicit_headers` 时只按原样构建。代码本身有错时每一级都失败，报告的是原样构建的错误，代价是多编一次 `slim`（最多约 15 秒；按约 6% 的编译失败率，平均每个 kernel 不到 1 秒）。`main.cpp` 用私有预编译头：按 torch 为 `main.cpp` 生成的编译参数（去掉 `-DTORCH_EXTENSION_NAME`）、编译器和 torch 版本，把 `<torch/extension.h>` 预编译成 `.gch`，经 `-include` 引入。每组参数只建一次（约 23 秒、0.7 GB），用 `-Winvalid-pch -H` 确认 gcc 确实采用后才启用；建不成的组合记下不再尝试，别的进程正在建时不等待，任何异常都退回普通编译。缓存在 `$SPECGEN_PCH_DIR`（默认 `~/.cache/specgen/pch`），不写共享的 torch 安装目录（torch 自带的 `use_pch` 会写进去）。

  实测（4 个构建并发）：新提示生成的 10 个 kernel 构建中位数 27.9 秒 → 15.6 秒，单次构建的 CPU 时间约 38 秒 → 20 秒；旧提示生成、几乎都引入 `<torch/extension.h>` 的 8 个 kernel 46.9 秒 → 15.1 秒；此前能编过的 48 个真实 kernel 都能用 `slim` 编过，GPU 验证结果与原样构建一致。剩下的约 15 秒是 nvcc 在宿主和设备两遍编译中解析 ATen 的 `Tensor` 声明，nvcc 不支持预编译头。把 `torch::Tensor` 的包装从 `.cu` 移到 `cpp_sources` 属于改写代码结构，不能可靠地自动完成，所以由 system prompt 要求模型这样写，再由 `bare` 一级自动识别。

  system prompt 同时说明 `cuda_sources` 不需要 `<torch/extension.h>`，并要求代码自己引入用到的头文件（如 `getCurrentCUDAStream` 需要的 `<ATen/cuda/CUDAContext.h>`、`CUDART_*` 常量需要的 `<math_constants.h>`）：这两类漏引入是编译失败最常见的原因，`<torch/extension.h>` 也并不提供它们。
  system prompt 还要求每个 block 超过 48 KB 的共享内存用动态共享内存，并在启动前调用 `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`；方案表里 illegal memory access 的固定提示也写上这一条。B300 本地冒烟时，`level3/50_ReLUSelfAttention` 第 0 轮 74 个 kernel 有 62 个在验证时报 illegal memory access，其中 57 个用了动态共享内存却没调用它（主流推理的设计漏了，spec 都继承了）；补上这一行后主流 kernel 不再崩溃。加了这条规则再跑一次，同一任务的崩溃由 62 个降到 18 个。
- **Validator**：只验证正确性，多组随机输入比对参考输出，默认容差 `atol = rtol = 1e-2`（TF32、tensor core 累加顺序带来的误差约在 1e-3 量级）。每个输出张量除了逐元素满足容差，整体的相对误差 ‖生成 − 参考‖₂ / ‖参考‖₂ 也必须不超过 `rtol`：输出数值很小时 `atol` 形同虚设，例如列数 393216 的 softmax 输出约 3e-6，只算了 1/16 行、其余行留着旧内存的 kernel 也能逐元素通过（实测拿到了 25× 的“加速比”）。生成模型运行前，先按参考输出的形状申请显存、填满 NaN 再释放，缓存分配器会把这些块分给生成模型的输出，没写到的元素就是 NaN，不会碰巧等于旧数据。常规 trial 之后再做一次分布偏移的 trial（`--validation-shift-trial`，默认开）：浮点输入原地变成 2x − mean(|x|)/2，范围扩大一倍、非负输入也会出现负值，用来拒绝写死了 `get_inputs()` 取值范围的 kernel（实测一个按 x ∈ [0,1) 写死量化比例的 INT8 GEMM 在原分布下误差 0.44%，换分布后 25–72%）；参考实现在这组输入上报错或给出 NaN/Inf 时（log、sqrt 之类）跳过。验证始终在分配的 GPU 上进行：输入直接在 GPU 上生成，输出在 GPU 上分块比较；子进程拿不到 CUDA 时报 `cuda_unavailable`，不退到 CPU。每卡可并发多个（`--validations-per-gpu`），OOM 时退回独占重试一次。参考输出每个任务只算一次并缓存。评测子进程固定 `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`：NVIDIA 的 PyTorch 容器默认把它设成 1，参考实现里的 FP32 矩阵乘就会走 TF32（B300 上 4096 方阵 2.09 ms → 0.157 ms），基线随机器而变。
- **Profiler**：
  - 计时：CUDA event 多次测量取中位数，独占一张卡，结果（latency、speedup）用于选优和终止判断。
  - ncu：只采之后的 prompt 会展示的 kernel（每轮最多 `--ncu-top-n` 个）：新的历史最优；topk 模式再加本轮最快的几个；summary 模式再加每类方案的新最优（表里那一类的 ncu 数据来自它）；不开 spec 时就是主 kernel。有更快的出现就插到前面，属于"有空闲才做"的任务。本轮结束时只保留下一轮 prompt 要展示的 kernel 的 ncu，下一轮开始前最多等 `--ncu-wait-s` 秒。没有 GPU 性能计数器权限（`ERR_NVGPUCTRPERM`）时返回 `ncu_unavailable`，本次运行自动停用 ncu。

### 资源池（全局共享）

- 实现为一个中央调度器：每台设备（GPU 或 CPU 编译池）有角色和容量；任何状态变化（提交、完成、取消、角色调整）后按顺序尝试放置所有排队任务，不同角色之间不会互相阻塞。
- 优先级从高到低：计时、验证、ncu。同类中主生成的 kernel 先于 spec，其余先来先服务；ncu 之间按 speedup 从快到慢。
- 独占任务（计时、ncu、OOM 后的独占验证）找不到空闲设备时预约一台：预约期间该设备不再接新任务，原有任务结束后由预约者使用。
- 三种模式（`--gpu-allocation`）：
  - `fixed`：两池卡数固定（`--num-validator-gpus`、`--num-profiler-gpus`）。
  - `dynamic`（默认）：按验证负载和计时负载的比例分卡，每池至少 1 张；只挪空闲卡，带迟滞；ncu 不计入负载。负载 = 排队加在跑的任务数 × 该类任务的平均耗时（验证再除以每卡并发数）；每 10 秒检查一次，总负载不足 60 秒时不调整，两次调整至少间隔 30 秒，每次只挪一张卡。
  - `fifo`：任何卡都能接任何任务；计时和 ncu 先预约一张卡（停止往该卡派新的验证，等它空出来再跑）。
- ncu 同时占用的卡数默认不超过计时卡数减一，保证新来的计时任务不会被长时间的 ncu 挡住（任务不可抢占）。

### 推测控制

- **按需调度**（`--spec-schedule demand`，默认；`control/pacing.py`）：瓶颈是 profiler（计时和 ncu 独占），编译和验证跟得上，所以 spec 的节奏由 profiler 决定，每次只发 1 个、均匀分布在整段推理上。每来一段推理增量就判断一次：

  \[ \text{waiting} + p_{pass} \cdot \text{upstream} < \text{target} = \max(s + \text{extra},\ s \cdot n),\quad s = \frac{\text{profiler 卡数}}{\text{在跑任务数}} \]

  extra 默认 1（`--spec-target-extra`），n 默认 1.5（`--spec-target-factor`）。waiting 是本任务排队中的 profiler 工作（按计时任务折算，一个 ncu 任务记为 ncu 平均耗时 / 计时平均耗时），upstream 是本任务还在生成、编译或验证的 spec，\(p_{pass}\) 是本任务已结束的 spec 里走到计时的比例（先验 0.4，权重 2）。满足时用当前推理前缀发一个 spec；两次之间至少隔 `--spec-min-gap-tokens`（500）个推理 token，第一次至少在 `--spec-first-tokens`（500）之后；每轮默认不设上限（`--spec-max-per-iteration`）；`--max-inflight-spec-gen` 仍限制 LLM 并发。因为约六成 spec 会失败，这个条件自然让上游保持几个在准备，profiler 空下来时有现成的 kernel。
  - **轮内反馈**：新 spec 的推理前缀之后、`</think>` 之前附上本轮已结束候选的测试结果，每个一行（自报设计、正确且多少倍或错误摘要），始终包含本轮最快的那个及其 ncu 摘要（和方案表的 ncu 列同一格式），最多 12 行、2400 字符；同时写入 kernel 目录的 `spec_notes.txt`。反馈放在所有请求共享的前缀之后，不影响缓存命中。
  - **后进先出**：spec 的编译、验证、计时任务按发出顺序倒序排队（越新越先），主 kernel 和确认计时始终在前。
  - **达到终止标准时等 ncu**：某个 kernel 满足终止条件时（含余量；未开启终止时为超过目前的最佳且不低于参考实现），它的 ncu 和计时同档、排在 spec 计时之前，同时暂停发 spec，直到 ncu 出结果、等满 `--best-ncu-wait-s`（30 秒）或这一轮结束：确认计时通过就结束本轮，没通过则接下来的 spec 能看到它的瓶颈分析。只是成为本轮最快、没达到标准的 kernel 不暂停，其他 ncu 都不阻塞。暂停与恢复记为 `spec_paused` / `spec_resumed` 事件。
- **TriggerMonitor**（`--spec-schedule triggers`；`configs/triggers.yaml`，关键词只用英文、正则、不区分大小写）：
  - 命中后等待 `after_tokens`（默认 500）再截取推理前缀；等待期间不重复触发。
  - 两次触发之间有最小间隔，每轮触发次数有上限。
  - 距上次触发超过 `fallback_tokens`（默认 2000）仍未命中则兜底触发。
  - 推理结束（开始输出正文）后停止触发。
- **SpecBudget**（全局一个）：评测流水线容量

  \[ C = \min\left(\frac{G_v\, s}{t_v},\ \frac{G_p}{r_{valid}\, t_p}\right) \]

  其中 \(G_v, G_p\) 为两池卡数，\(s\) 为每张验证卡的并发数，\(t_v, t_p\) 为（该并发度下）验证和计时的平均耗时，\(r_{valid}\) 为验证通过率。每个任务按公平份额分得 \(C\) 的一部分；设 \(H\) 为距下一次能发 spec 的预计时间加一次 spec 生成耗时，\(W\) 为系统中已有工作量（排队数 + 在跑数 + 在途 spec 数 × \(r_{parse}\)），则

  \[ k = \mathrm{clip}\left(\left\lceil \frac{C\,H - W}{r_{parse}} \right\rceil,\ k_{min},\ \min(k_{max},\ \text{剩余并发})\right) \]

  全局在途 spec 数受 `--max-inflight-spec-gen` 限制（默认不限，自部署 vLLM 时建议设置）；API 限流时退避重试。
  在途 spec 只计 LLM 请求阶段：名额在生成结束时归还，编译和评测由上式中的 \(W\) 计入。名额用满时（`--spec-queue`）：`drop` 让 k 不超过空闲名额；`replace` 让超出的 spec 排队，同一任务的新触发取消其仍在排队、尚未开始生成的旧 spec 并重新计算 k（新前缀包含旧前缀）。
  实现细节：容量里还包括 CPU 编译池 \(G_b / t_b\)；各类任务的平均耗时取资源池的滑动平均，没有数据时用先验值（编译 50 秒、验证 20 秒、计时 15 秒、spec 生成 30 秒、推理 50 token/s、通过率 0.5、解析率 0.8）；k 在决定时就预留在途名额，每个 spec 结束时释放。算出 k = 0（流水线已满或在途名额用完）的触发不计入每轮触发次数上限。`--spec-k N` 改为固定 k。
- 同一前缀的 k 个 spec 使用比主生成更高的采样温度；进入编译前按代码内容去重。

### 迭代结束

- **baseline**（`--no-enable-spec-gen`）：传统的逐轮迭代优化，对照实验都以它为基准。每轮只有主生成写出的一个 kernel，评测完这一轮就结束，没有 spec，也没有提前终止（这种模式下 `--enable-termination` 被忽略并给出警告，`config.json` 的 `resolved.termination` 为空）。HTML 报告里 baseline 只列实际运行一行，不区分终止条件。
- **提前终止**（`--enable-termination` + `--termination-criteria`，只在开 spec 时生效）：任一 kernel 的计时结果满足条件即结束本轮。
  - `first_valid`：本轮第一个通过验证并完成计时的 kernel。
  - `best`：speedup 超过历史全局最优。
  - `hist_avg`：speedup 超过之前各轮胜出 kernel 的平均值。
  - 第 0 轮以参考实现（1.0x）为基准；`best` 的基准不低于 1.0x（历史最优比 PyTorch 还慢时，比 PyTorch 慢的 kernel 不结束迭代——否则每轮只要比一个慢 kernel 快一点就终止，主生成一再被中止）；`--termination-margin` 表示至少要超出的比例（默认 0.03，计时噪声约 ±1%）。
  - `--termination-confirm N`：kernel 第一次满足条件后再计时 N−1 次，每次都满足才结束迭代，防止一次偏高的计时（噪声约 ±1–2%）中止主生成；复测过的 kernel 以几次计时的平均值作为加速比参与选优。复测的结果写在 `benchmark_confirm<n>.*`。
  - 结束时：epoch 加一，中止主生成的流和在途 spec 请求，丢弃旧 epoch 的排队任务，杀掉在跑的验证和计时；下一轮 prompt 入选 kernel 的 ncu 保留、不等待，结果写回历史；触发终止的 kernel 即本轮选中的 kernel。
- **自然结束**：主 kernel 评测完后，其余候选最多再等主生成时长 × `--grace-ratio`（默认 0.05），然后在本轮有效 kernel 中选最快的；一个有效的都没有，就把主 kernel 的错误信息反馈给下一轮。被宽限期截断的候选状态记为 `grace_truncated`，未完成的 spec 任务数写入迭代记录。配置里 `grace_ratio` 为空时等所有候选完成（测试用）。
- **epoch**：每个任务一个递增编号，所有排队和在跑的任务都带上它；旧编号的结果只记录、不参与选优。
- **续跑**（`--resume --run-id <已有运行> --max-iterations N`，其余参数照原运行给）：在原目录接着跑到第 N 轮。任务列表必须和原 `config.json` 一致；原配置不覆盖，本次配置写到 `config.resume<k>.json`，`run_started` 事件带 `resumed` 和 `previous_end`。每个任务按记录重建历史：`kernels.jsonl` 里已完成各轮的 kernel（代码取 `kernel.py`，没有的——静态检查拒绝、重复——按生成时同样的规则从 `content.txt` 取；计时取记录；ncu 分析取 `ncu.json`；错误信息按运行时同样的方式取自失败那一步的结果文件），`iterations.jsonl` 里每轮选中的 kernel，去重指纹和 spec 通过率；从第一个没完成的轮次开始（没跑完的那一轮重跑：它留下的目录改名为 `iter_NNN.interrupted<k>`，它的 kernel 记录移到 `kernels.interrupted<k>.jsonl`，不进历史）。`events.jsonl`、`kernels.jsonl`、`iterations.jsonl` 接着追加；分析工具把两段之间的空闲时间从时间线里去掉，HTML 在续跑处画一条标记；上一段没结束的评测任务在它最后一个事件的时刻收尾（`interrupted`），没跑完那一轮的主流、spec、触发和计时不进分析（和记录一致），它的时间段、发起的 spec 数和主流已推理的 token 记在 `RunData.interrupted`（花了但不在 token 统计里）。用今天的 sqmm 运行核对过：从第 1、3、5、7、9 轮恢复后重建的 prompt 与当时的 prompt 逐段一致。

## 3. Prompt 布局（保护 KV cache）

前缀缓存只能复用完全相同的开头，因此按稳定程度从高到低排列：

| 位置 | 内容 | 什么时候会变 |
|---|---|---|
| system | 角色和输出规则（所有轮次共用一份） | 不变 |
| user ① | GPU 参数 | 整次运行内不变 |
| user ② | 任务说明和参考实现 | 同一任务内不变 |
| user ③ | 历史最优 kernel（代码、计时、ncu 指标和 ncu 分析）；标题注明它已测过、不要原样提交、要尽力尝试新的优化 | 只在刷新最优时变 |
| user ④ | 本轮反馈：上一轮最快的 `--feedback-top-k`（默认 3）个 kernel（不含历史最优），或主 kernel 的错误信息（`--history-prompt topk` 和 `topk+summary`） | 每轮都变 |
| user ⑤ | 已尝试方案表（`control/attempts.py`，见下）：`--history-prompt summary` 时取代 ④，`topk+summary`（默认）时与 ④ 并存 | 每轮都变 |

**已尝试方案表**（只用规则，不调 LLM）：每个 kernel 第一行自报 `# APPROACH: …`（续写模式的 spec 由预填强制写出）；规则从去掉注释的代码里读出精度（int8/fp8/bf16/tf32/fp16/fp32，按指令和类型判断，TF32 WMMA 不会被当成 fp16）、计算方式（wgmma/mma.sync/wmma/调库/SIMT）和关键技术（cp.async、TMA、ldmatrix、split-K、warp shuffle、向量化、原子操作），特征相同的 kernel 归为一类。每类一行：特征、该类试过的 tile 与流水级数组合（从自报里读出，如 `128x128x64, 4 stages: 21/19 1.595x` 表示试了 21 次、通过 19 次、最快 1.595x；有上限的表每类列 4 个，不限时全列）、其最优 kernel 的自报设计、尝试数、通过数、最高加速比、由数据判定的状态（当前最优 / 全部失败 / 比参考实现慢 / 明显低于最优）。失败按归一化后的报错（nvcc / ptxas `error:` 和 `fatal:` 行、链接错误、`CUDA error:`、异常行、数值错误的种类，引号内容、路径、符号和数字替换掉）聚类，列出次数、示例、涉及的方案和 tile 组合，常见错误附一句固定提示；验证时才编译失败的（`Error building extension`）算编译错误，并取出其中的编译器报错。上限：`--summary-max-approaches`（多出的合并为一行）、`--summary-max-failures`、`--summary-max-chars`，默认都是 0（不限，即完整的表）。

**ncu profile**（`evaluation/ncu.py`）：一个模块的一次 forward 可能启动好几个 CUDA kernel（例如转换 + GEMM）。驱动程序（`ncu_harness`）把每次 forward 包在 NVTX 区间 `forward<i>` 里，ncu 用 `--nvtx --nvtx-include forward3/` 只采第 `--ncu-forward`（默认 3，已预热）次 forward 里的全部 kernel，每次启动采一次（一次 forward 里调用两次的 kernel 就采两次，占比因此准确），采 `--ncu-sections`（默认 SpeedOfLight、Roofline、Compute/Memory Workload、LaunchStats、Occupancy、SchedulerStats、WarpStateStats、InstructionStats、SourceCounters）和 `--ncu-metrics`；NVTX 没采到时改为每个 kernel（按 mangled 名）采第 3 次调用。报告用 `--page raw --csv --print-units base` 逐次启动读入（第二行是单位，统一为纳秒、字节等基本单位），按 kernel 分组：每个 kernel 有自己的指标（只在自己的启动之间平均）、在这次 forward 里的耗时占比和规则结论（`--page details`），不同 kernel 绝不合并。单次超过 `--ncu-timeout-s`（120 秒）就杀掉整个进程组，改用 SpeedOfLight、LaunchStats、Occupancy 三部分重跑（结果标 `degraded`）。prompt 里每个带 ncu 结果的 kernel 后面附"Nsight Compute profile of one forward pass"：占时间不低于 5% 的 kernel（最多 3 个）各列吞吐（SM、内存、Tensor 管线——按活跃周期，tcgen05 和 mma.sync 都算，tcgen05 会注明）、内存（DRAM、L1、L2、DRAM 带宽）、占用率（实际、理论、受什么限制、寄存器、共享内存、block 和 grid 大小）、主要 warp 停顿原因，再列 ncu 的瓶颈总结和按预估加速比排序的 OPT/WRN 建议（主 kernel 最多 6 条，其余 3 条，去掉指向 ncu 界面和文档的句子）；不显示 ncu 锁频下测的耗时，多 kernel 时显示占比。已尝试方案表每行多一列 ncu：该类最优 kernel 的主要数据和 ncu 建议名（带预估加速比）。下一轮建 prompt 前最多等 `--ncu-wait-s` 秒，让要展示的 kernel 的 ncu 跑完（最后一轮之后也等一次）；等满上限、单次超时或降级会记 `ncu_alert` 事件并写日志。本轮结束后才完成的 ncu 不在 `kernels.jsonl` 里，见 kernel 目录的 `ncu.json` 和 `kernel_ncu` 事件。

**合规检查**（`control/compliance.py`）：kernel 必须手写，不得调用库 GEMM / 卷积（`torch.matmul`、`torch.mm`、`F.linear`、`@`、`at::matmul`、cuBLAS、cuDNN 等），连作为退路也不行，也不得修改 torch 的矩阵乘精度或 TF32 设置。两道检查：构建前用 `ast` 读 Python 代码（调用、赋值、`@` 运算符；自己扩展模块里的同名函数不算）并扫描 C++/CUDA 源码字符串；验证通过后，验证任务用 `torch.profiler` 记录一次 forward 实际执行的 kernel（不需要计数器权限，记在 `forward_kernels`），其中出现 cuBLAS / cuBLASLt / CUTLASS / cuDNN 的 kernel 也判为不合规。不合规的状态是 `forbidden`，不编译（静态）或不计时（运行时），进已尝试方案表的失败列表。

只有注释、没有代码的模块（例如只写了 `# APPROACH:` 一行就关掉代码块的 spec）记为 `parse_error`，不编译也不验证。
| user ⑤ | 固定的收尾指令 | 不变，但位于变化内容之后 |

- ①到③中不出现轮次号、时间戳、运行 ID 等会变的内容。
- 格式确定：JSON 键排序、数值固定精度、指标顺序固定、kernel 按 speedup 排序且平局规则固定、截断规则固定。
- reprompt 模式的 spec 请求把推理前缀追加在 ⑤ 之后；Anthropic 后端在 ② 和 ③ 之后设置缓存断点。

## 4. 接口（示意）

```python
class Task(Protocol):  # KernelBench 是其中一个实现
    task_id: str
    def reference_source(self) -> str: ...

class ChatBackend(Protocol):
    def stream_chat(self, messages: list[dict], params: ChatParams) -> AsyncIterator[GenEvent]: ...

class Generator(Protocol):
    async def generate_main(self, prompt: Prompt, sink: GenerationSink) -> GenerationResult: ...
    async def generate_spec(self, prompt: Prompt, reasoning_prefix: str, *, seed: int) -> GenerationResult: ...

class Evaluator(Protocol):  # 本机子进程或远程 worker
    async def run(self, job: EvalJob, slot: GpuSlot) -> dict: ...

class ResourcePool(Protocol):
    async def submit(self, job: PoolJob) -> Any: ...  # PoolJob 带 kind / priority / task_id / epoch
    def cancel(self, task_id: str, before_epoch: int, *, kill_running: bool) -> int: ...
    def stats(self) -> PoolStats: ...

class TriggerMonitor(Protocol):
    def feed(self, reasoning_delta: str, total_tokens: int) -> Trigger | None: ...

class TerminationCriterion(Protocol):
    def satisfied(self, rec: KernelRecord, history: History) -> bool: ...
```

## 5. 配置文件

`configs/models.yaml`（自部署端点为占位符；DeepSeek API 已按官方文档填好）：

```yaml
DeepSeek-V4-Flash:
  tokenizer: generic
  reasoning: field
  reasoning_field: reasoning_content
  sampling: {temperature: 1.0, top_p: 0.95, spec_temperature: 1.0, max_tokens: 131072}
  endpoints:
    selfhost: {backend: vllm, base_url: "${SPECGEN_DEEPSEEK_FLASH_SELFHOST_URL}", model: "<served-model-name>"}
    api:
      backend: api
      base_url: https://api.deepseek.com
      api_key_env: SPECGEN_DEEPSEEK_API_KEY
      model: deepseek-flash
      modes:
        main: {thinking: {type: enabled}, reasoning_effort: max}
        spec: {thinking: {type: disabled}}
```

`configs/triggers.yaml`（第 2 步加入）：

```yaml
case_sensitive: false
fallback_tokens: 2000
default_after_tokens: 500
min_gap_tokens: 500
max_triggers_per_iteration: 20
groups:
  - name: memory_hierarchy
    patterns: ['__shared__', '\bshared memory\b', '\bsmem\b', '\bbank conflicts?\b']
  - name: tiling
    patterns: ['\btile[ _-]?size\b', '\bTILE_[A-Z]+\b']
  - name: code_start
    patterns: ['__global__', '```(cuda|cpp|python)']
    after_tokens: 300
```

## 6. 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model` | `GLM-5.1-FP8` | 对应 `models.yaml` 中的一项 |
| `--endpoint` | `selfhost` | `selfhost` / `api` |
| `--tokenizer` | `generic` | token 计数用的分词器 |
| `--gpus` | 全部可见卡 | 评测用的卡（排除 LLM 服务占用的卡） |
| `--gpu-allocation` | `dynamic` | `fixed` / `dynamic` / `fifo` |
| `--remote-worker` | 无 | `specgen worker` 的地址，可重复；token 放在 `$SPECGEN_WORKER_TOKEN` |
| `--cuda-arch` | 使用 GPU 时必填 | kernel 编译架构：H100/H200 `9.0a`，B200 `10.0a`，B300 `10.3a`；`auto` 按 GPU 检测，`inherit` 沿用环境 |
| `--gpu-type` | 检测 | prompt 中的硬件参数表；检测不到时报错 |
| `--build-jobs` | `auto` | 单个 kernel 构建的 `MAX_JOBS`；`auto` 为有效 CPU 数除以编译槽位数（2 到 32） |
| `--atol`、`--rtol` | 1e-2、1e-2 | 正确性容差 |
| `--fast-cuda-includes` | 开 | 构建时依次尝试不含 torch 的 `.cu`（只在源码不用 torch 时）、精简前导、去掉整套 torch 头文件、原样，编不过自动回退 |
| `--build-pch` | 开 | `main.cpp` 使用私有预编译头（`$SPECGEN_PCH_DIR`，默认 `~/.cache/specgen/pch`，每组编译参数约 0.7 GB） |
| `--num-validator-gpus`、`--num-profiler-gpus` | 各一半 | fixed 下固定，dynamic 下为初始值 |
| `--validations-per-gpu` | 8 | 每张验证卡的并发数 |
| `--build-workers` | `auto` | CPU 编译池大小：有效 CPU 数 / 2，最多 64；0 表示在验证时编译 |
| `--enable-spec-gen` / `--no-enable-spec-gen` | 开 | 推测生成开关；关闭即 baseline（逐轮迭代，无 spec、无提前终止） |
| `--trigger-config` | 内置 | 触发配置文件 |
| `--trigger-after-tokens`、`--fallback-trigger-tokens` | 500、2000 | 覆盖配置文件中的值 |
| `--spec-k` | `auto` | 每次触发的 spec 数量：`auto` 由 SpecBudget 决定，或固定值 |
| `--spec-k-min`、`--spec-k-max` | 1、4 | k 的范围（也限制固定的 `--spec-k`）；同一批越往后温度越高，第 5 个起几乎不通过验证 |
| `--max-inflight-spec-gen` | 不限 | 全局同时在途的 spec LLM 请求上限 |
| `--spec-queue` | `drop` | 名额用满时：`drop` 少发；`replace` 排队并让新触发取代同任务未开始的旧 spec |
| `--spec-prompt-mode` | `auto` | `prefill` / `reprompt` / `auto` |
| `--spec-schedule` | `demand` | `demand`：按 profiler 需要逐个发 spec，附轮内测试结果；`triggers`：每次关键词触发发 k 个（以下几项只对 `demand` 有效） |
| `--spec-target-extra`、`--spec-target-factor` | 1、1.5 | 目标值 = max(本任务分到的 profiler 卡数 + extra, 卡数 × factor) |
| `--spec-min-gap-tokens`、`--spec-first-tokens` | 500、500 | 两次发 spec 之间、第一次之前至少的推理 token 数 |
| `--best-ncu-wait-s` | 30 | 有 kernel 达到终止标准时为等它的 ncu 暂停发 spec 的最长时间，0 表示不等 |
| `--spec-max-per-iteration` | 0 | 每轮每个任务最多发多少个 spec，0 表示不限 |
| `--ncu-top-n` | 6 | 每轮最多采几个 kernel（只采之后的 prompt 会展示的），0 表示关闭 |
| `--ncu-sections` | 内置列表 | 采哪些 ncu 部分并把其规则结论放进反馈；`none` 只采指标 |
| `--ncu-forward` | 3 | 采第几次 forward 里的全部 kernel（每次启动一次） |
| `--ncu-launch-count` | 64 | 这次 forward 里最多采多少次 kernel 启动 |
| `--ncu-timeout-s` | 120 | 单次 ncu 超过就杀掉，改用最小分析部分重跑；0 不限 |
| `--ncu-wait-s` | 300 | 建下一轮 prompt 前，最多等要展示的 kernel 的 ncu 多少秒；0 不等 |
| `--feedback-top-k` | 3 | topk 的 k：下一轮 prompt 放入上一轮最快的几个 kernel |
| `--history-prompt` | `topk+summary` | 除历史最优代码外放什么：`topk` 上一轮最快的 kernel 代码，`summary` 已尝试方案表，`topk+summary` 两者都放 |
| `--summary-max-approaches`、`--summary-max-failures`、`--summary-max-chars` | 0、0、0 | 方案表的上限，0 表示不限 |
| `--enable-termination` | 关 | 提前终止开关（只在开 spec 时生效；baseline 忽略） |
| `--termination-criteria` | 开启时必填 | `first_valid` / `best` / `hist_avg` |
| `--termination-margin` | 0.03 | 至少要超出基准的比例 |
| `--termination-confirm` | 2 | 满足条件的 kernel 一共计时几次（每次都要满足；`first_valid` 不复测） |
| `--validation-shift-trial` | 开 | 验证多做一次分布偏移的 trial |
| `--grace-ratio` | 0.05 | 自然结束时最多等待主生成时长的比例 |
| `--max-iterations` | 10 | 最大迭代轮数 |
| `--resume` | 关 | 在 `--run-id` 指定的已有运行上续跑到 `--max-iterations` 轮 |

## 7. 记录与目录

PerfRecorder 将所有事件追加写入 `events.jsonl`（生成开始/结束、任务排队/开始/结束、GPU 角色变化、触发、终止），并派生：

- **kernel 记录**（`kernels.jsonl`）：迭代号；来源（main / spec）；spec 的触发原因、前缀截止的 token 位置、本批 k 和批内序号；生成耗时和 token 数；能否解析；验证的排队、编译、运行耗时、所在 GPU、同卡并发数、是否通过及详情；计时的排队和运行耗时、latency、speedup；ncu 是否采集、排名、耗时和指标；是否被选中、是否触发终止、是否被宽限期截断或因 epoch 变化被取消。
- **迭代记录**（`iterations.jsonl`）：结束原因、主生成的 token 数和速率、触发次数和每次的 k、选中的 kernel。
- **资源时间线**：每张卡的忙闲区间和角色变化。

运行目录必须是新建的（原子 `mkdir`），同一个 run id 不能被两个进程同时或先后复用。

```text
<output_root>/<run_id>/
  config.json  events.jsonl  kernels.jsonl  iterations.jsonl
  <task>/ref_cache/                       参考输出缓存
  <task>/iter_000/
    prompt.json  main/{reasoning.txt, content.txt}
    kernels/<kernel_id>/{kernel.py, content.txt (spec), build.json, validate.json, benchmark.json, ncu.json,
                         *.request.json, *.log, torch_extensions/}
```

kernel id：主 kernel 为 `i000-main`，spec 为 `i000-t03-s1`（第 0 轮第 3 次触发的第 1 个 spec）。

## 8. 远程 worker

`specgen worker --gpus 0,1 --cuda-arch 9.0a --port 8765` 在另一台机器上提供评测服务（HTTP + JSON，可选 token）：

- `GET /info`：GPU 列表、编译架构、GPU 型号。
- `POST /jobs`：请求带任务和 kernel 的源码、作业参数和槽位；worker 把源码写到自己的目录（同一个 kernel 的工作目录固定，编译产物可在验证和计时之间复用），用同样的隔离子进程执行并返回结果。参考输出缓存使用 worker 自己的目录。
- `POST /jobs/<id>/cancel`：客户端取消时杀掉对应子进程。

`specgen run --remote-worker http://host:8765` 启动时读取各 worker 的 GPU，按同样的分配方式加入资源池（设备带 host 标记），评测时按槽位路由到本机或对应 worker。本机的 CPU 编译池仍会先编译一次，用来提前排除编译错误；远端验证时在远端重新编译。

## 9. 分析工具

- `specgen summarize RUN_DIR [--json]`：每个任务每轮的结束原因、选中的 kernel 及来源、主 kernel 状态、触发数和 spec 数、本轮第一个计时结果出现的时间和来源；spec 的解析、通过、计时、被选中次数；各类任务的排队和执行时间（均值、p90）；GPU 利用率。
- `specgen report RUN_DIR [-o report.html]`：自包含的 HTML 时间线（主生成和触发点、spec 生成、每个槽位上的编译/验证/计时/ncu、迭代边界和终止线、GPU 角色调整），可按时间回放，显示触发点的推理片段（关键词高亮）、事件流和加速比随时间的折线。`--compare OTHER_RUN [--label A --compare-label B]` 再加对照：对照运行自己的时间线（画在下面，共用同一时间轴），两次运行各任务的最佳加速比随时间的阶梯线（本运行红色、对照运行蓝色），以及完成轮数、用时、最终最佳、达到对方最终最佳的时刻、评测和通过的 kernel 数、主生成被提前中止的轮数，用来比较开 / 不开 spec 等配置。两条时间线的资源行一样多：每种角色按两次运行里用到的最多槽位数画（`report.slot_rows`），资源相同的运行不会因为用得少而显得资源少。终止条件表每行另有 profiler GPU 利用率：profiler 卡上有计时或 ncu 任务在运行的时间（重叠只算一次）占该行保留时间的比例。一页可以放多个运行（`report.render_page`，每个运行一节，各自回放），例如几个任务的 SpecGen 对 baseline；每节有一个可收起、展开的标签栏，正在看的那一节的标签栏一直置顶。有对照时，本运行的数值后面括号里给出相对对照运行实际运行的值：用时和 token 是占对照运行全程的百分比（如 33.91% of Baseline），加速比是对照运行最终最佳的倍数（如 3.87x of Baseline）。终止条件的估算可以再加预算（`criteria.simulate(..., budget=("seconds" 或 "tokens", 量))`）：在截断后的时间线上，用到给定的时间或等价 token（`criteria.equivalent`：命中缓存的输入 × 1/50 + 未命中的输入 + 4 × 输出，即按未命中输入的价格折算）就停在不超出预算的最后时刻，用来在和 baseline 相同的时间或花费下比较。时间线上因超时被终止的评测任务画红框，以免长条被误认为别的东西。续跑时换了 GPU 的运行可用 `data.align_resumed_devices` 把后面各段的设备按角色改成第一段的编号（只改标签），时间线上每个槽位仍是一行。

## 10. 实现顺序

0. 探测脚本（`tools/probe_stream.py`）：确认各模型的流式格式、token 用量上报方式、能否前缀续写，据此填写模型配置。
1. 骨架：配置、KernelBench 任务、ModelProfile、流式 Generator、Validator/Profiler 子进程执行、fixed 资源池、PerfRecorder、不带 spec 的迭代循环。
2. spec：TriggerMonitor、固定 k、两种 spec prompt 模式、代码去重。（已完成）
3. 自适应：全局 SpecBudget、dynamic 和 fifo 资源池、Builder、验证并发、参考输出缓存。（已完成）
4. 终止：criteria、epoch 取消、宽限期。（已完成）
5. 其余：其他 LLM 后端、远程 worker、分析脚本，移除旧的 OptKernel 脚本。（已完成）
