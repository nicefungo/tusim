# Implementing a Tensor Unit (TU) C‑Model: Necessity, Impact, and Practical Roadmap

## Executive Summary

Tensor Units (TUs)—the matrix/tensor compute blocks embedded in modern NPUs and AI accelerators—are central to the performance and correctness of deep learning chips. A TU C‑model ("cmodel") is a software implementation of the TU in C/C++ (or similar), used as a golden functional reference and often as a high‑speed simulation back end for architecture exploration and software co‑development. This report analyzes whether building such a cmodel is necessary and worthwhile, how difficult it is in practice, and how it integrates into contemporary chip design flows. It concludes with a concrete, staged implementation plan aimed at a modern AI accelerator project.[^1][^2]

The research suggests that a TU cmodel is **highly valuable and usually essential** for non‑trivial AI accelerator projects, but the exact required fidelity (purely functional vs. cycle‑accurate) and scope (TU‑only vs. whole NPU) depend on project stage, team size, and verification goals. A purely functional, bit‑accurate TU cmodel delivers disproportionate benefits for algorithm validation, RTL verification, and compiler bring‑up at relatively modest complexity, while full cycle‑accurate system models such as NPUsim, SCALE‑Sim, NeuSim, and ONNXim demonstrate how far the concept can scale when performance and system effects also matter.[^3][^4][^5][^6][^7][^8][^9][^10][^1]


## 1. Background: C‑Models and TU Context

### 1.1 C‑Based Functional Models in Hardware Design

In ASIC/FPGA design, C‑based verification refers to using C or C++ to build reference models, stimulus generators, and checkers for hardware designs, particularly at the functional level. These models serve as **golden references** whose outputs are treated as the specification when validating RTL implementations. They are typically bit‑accurate (matching arithmetic results exactly) and sometimes cycle‑accurate when timing behavior is important.[^6][^1]

C‑models are widely used because they run orders of magnitude faster than RTL simulations, can be easily integrated with software stacks, and are portable across simulators via DPI‑C or SystemC interfaces. For algorithmic accelerators such as codecs or AI compute blocks, C‑models allow hardware and software teams to iterate on algorithms and microarchitecture in parallel, reducing late‑stage bugs and specification drift.[^11][^12][^1][^6]

### 1.2 Tensor Units in Modern AI Accelerators

Tensor Processing Units (TPUs) and other NPUs dedicate large on‑chip systolic or SIMD arrays to high‑throughput tensor operations such as GEMM and convolution. These arrays, often called Tensor Units or Tensor Cores, implement matrix multiply‑accumulate (MMA) pipelines with local scratchpad memory and support for mixed precision (INT8, BF16, FP16, etc.). The TU is typically the **dominant contributor to compute throughput and energy consumption**, making its behavior critical both for correctness and for system‑level performance.[^5][^13][^14]

NPU simulators such as NPUsim, SCALE‑Sim, NeuSim, ONNXim, and EONSim demonstrate that detailed modeling of tensor compute and memory behavior is central to evaluating accelerator architectures and serving workloads. Although these tools often focus on entire NPUs rather than a single TU, they reinforce the idea that a high‑fidelity model of the tensor compute engine is foundational for meaningful analysis.[^7][^8][^9][^10][^3]


## 2. Necessity and Importance of a TU C‑Model

### 2.1 Role Across the ASIC/FPGA Development Cycle

C‑based reference models fit into the ASIC/FPGA lifecycle at multiple stages: specification, architecture, RTL design, RTL verification, and hardware–software validation. At the specification and architecture stages, a TU cmodel allows architects to encode the intended arithmetic semantics, dataflows, and precision behaviors in an executable form that can be validated with real workloads. During RTL design and verification, the same cmodel becomes the oracle to which RTL outputs are compared in co‑simulation environments, enabling immediate detection of functional deviations.[^12][^1]

After RTL is relatively stable, the TU cmodel remains useful for regression testing, as a backend target for compilers, and as a performance oracle in higher‑level simulators that use deterministic TU timing models (for example, ONNXim explicitly assumes deterministic compute latency for tiles processed from scratchpad). In this sense, the TU cmodel is not just a point solution for pre‑RTL exploration; it becomes a long‑lived artifact used throughout the chip’s lifecycle.[^15][^3]

### 2.2 Functional Verification and Bug Detection

Functional simulation is the process of testing and verifying that a chip or IP block behaves according to its logical specification, independent of physical timing. For complex AI accelerators, functional verification must check not only correct matrix arithmetic but also corner cases such as overflow, NaN/Inf propagation, denormal handling, quantization saturation, and sparsity semantics.[^2]

C‑models are ideal for this because they can encode exact numerical behavior and be exercised with large test suites generated from real neural network traces. NPU simulators such as NPUsim and EONSim use value‑aware functional simulation to analyze detailed accelerator behavior over a wide range of DNN architectures, demonstrating how functional models reveal nuanced interactions between operators, dataflows, and memory subsystems. A TU cmodel acts as the localized version of that capability, focusing on the tensor compute block but participating in the same verification ecosystem.[^10][^1][^2][^7]

### 2.3 Architecture Exploration and Performance Modeling

Architects use simulators like SCALE‑Sim, ONNXim, EONSim, NeuSim, and NPUsim to explore design points across array dimensions, scratchpad sizes, and memory hierarchies. These tools generally separate **functional correctness** from **timing/performance modeling**: they either use a simple functional core and elaborate memory and scheduling models, or they integrate validated performance models for matrix operations while modeling memory accesses in more detail.[^4][^8][^9][^3][^7][^10]

A TU‑level cmodel fits into this paradigm as the functional oracle and, optionally, as a local timing model. ONNXim, for example, assumes deterministic compute latency for tiles processed from scratchpad, implying that once the TU compute latency is parameterized, system‑level performance can be modeled by focusing on DRAM and NoC contention. Similarly, EONSim integrates a validated performance model for matrix computations together with detailed memory simulation for embeddings, enabling accurate inference‑time predictions with low error relative to TPUv6e. A carefully designed TU cmodel can therefore serve as the backbone of such performance models.[^3][^15][^7]

### 2.4 Software Co‑Development and Compiler Bring‑up

Software stacks for AI accelerators—compilers, graph schedulers, and runtime libraries—need a target whose semantics are fixed early enough to guide development. Because real hardware is unavailable during most of the software development cycle, teams rely on functional models and architectural simulators.[^1][^6]

A TU cmodel allows compiler teams to emit TU‑level instructions or micro‑ops and validate them against a known‑correct model long before silicon. Higher‑level NPU simulators such as NPUsim and NeuSim accept full network descriptions (often via ONNX or framework exports) and rely on underlying modeled compute units to estimate performance and resource usage. In such stacks, a TU cmodel becomes a plug‑in compute engine that can be swapped between different simulation frameworks while maintaining consistent semantics.[^9][^10][^1]

### 2.5 When a TU C‑Model May Be Overkill

While the benefits are significant, a TU cmodel is not always necessary. Synopsys and other EDA vendors note that creating hardware‑accurate C models for every custom block may be impractical due to the required expertise and effort in C++/SystemC modeling, verification of the model itself, and protocol integration. For small ASICs, simple or well‑understood compute blocks, or projects with aggressive schedules, teams may instead rely on MATLAB/Simulink models, high‑level framework code, or direct RTL testbenches without a fully featured C‑model.[^6][^11][^12]

Moreover, if an accelerator primarily reuses standard blocks (e.g., a licensed third‑party NPU or a general GPU core) and the team has little control over the microarchitecture, investing in a detailed TU cmodel may not yield sufficient incremental value over existing vendor simulators and software reference implementations. In such contexts, a TU cmodel could be considered non‑essential.[^6]


## 3. Difficulty, Scope, and Trade‑offs

### 3.1 Levels of Fidelity

TU cmodels can be categorized by fidelity:

1. **Purely functional, bit‑accurate**: Implements the exact arithmetic behavior of TU instructions (MMA, convolution, elementwise ops), including rounding, saturation, and quantization, but ignores detailed timing. This is typically the first and most cost‑effective tier and is sufficient for correctness verification and compiler validation.[^2][^1]
2. **Latency‑annotated functional**: Adds parametric latency estimates (e.g., cycles per tile) for use in higher‑level simulators like ONNXim and EONSim, which assume deterministic compute latency while modeling memory in full detail.[^7][^3]
3. **Cycle‑accurate**: Models internal pipelines, stalls, and utilization at the cycle level, similar to NPUsim’s full‑system cycle‑accurate simulations or SCALE‑Sim’s modular cycle‑accurate timing. This provides maximum fidelity for performance tuning but is more complex and slower to simulate.[^8][^4][^10]

Choosing a level of fidelity is a key architectural decision. Many successful projects adopt a staged approach: start with a bit‑accurate TU cmodel, then gradually add timing annotations or integrate with a separate cycle‑accurate performance simulator when needed.

### 3.2 Functional Complexity

The core functional complexity of a TU cmodel arises from the combination of supported numerical formats, dataflows, and fused operations. Systolic arrays designed for CNNs and DNNs may support output‑stationary, weight‑stationary, or row‑stationary dataflows, each with different scheduling and tiling strategies. Mixed precision (INT4/INT8/INT16/BF16/FP16/FP32) and structured sparsity (e.g., 2:4 sparsity) require careful modeling of accumulation ranges, scaling factors, and sparsity masking.[^5]

Open‑source projects illustrate this complexity. SAURIA, a CNN accelerator based on an output‑stationary systolic array, is parametric in array shape, local memory, and arithmetic formats (tested extensively with FP16). NPUsim supports a wide variety of DNN accelerator architectures, including TPU‑like arrays, and models execution with real DNN data, stressing the importance of generality. EONSim extends modeling to both matrix and embedding operations, reflecting the growing diversity of workloads on NPUs.[^16][^10][^7]

### 3.3 Modeling Effort vs. Automation

Building a high‑quality C‑model demands expertise in C++/SystemC modeling, verification, and hardware protocol semantics. To mitigate this, tools such as Synopsys’ Synphony Model Compiler and C2R aim to automatically generate hardware‑accurate C models from RTL or high‑level datapath descriptions, reducing manual effort and improving consistency. MATLAB/Simulink‑based flows can also generate SystemVerilog DPI components and testbenches directly from high‑level models, effectively providing a C‑model‑like reference without traditional hand‑coded C++.[^11][^12][^6]

Nevertheless, for a custom TU microarchitecture with novel features, hand‑crafted C‑models remain common in high‑end teams, particularly where fine‑grained numerical behavior must be controlled and iterated frequently during architecture exploration.


## 4. Influence on Modern Chip Design Practices

### 4.1 Enabling Full‑Stack Co‑Design

Modern AI hardware design increasingly follows a full‑stack co‑design philosophy: architectures are tailored to representative workloads, compilers co‑optimize tiling and scheduling with hardware characteristics, and system software coordinates multiple accelerators. In this context, TU cmodels are a linchpin connecting hardware and software.[^8][^9][^10]

Simulators such as NPUsim, SCALE‑Sim, and NeuSim accept DNN models and hardware configurations, then evaluate latency, bandwidth, energy, and utilization. These frameworks embed or rely upon functional models of tensor compute units whose behavior matches that of the physical hardware. By giving compiler and framework developers early access to an executable TU semantics, teams can co‑optimize dataflows, parallelism configurations, and model architectures long before tape‑out.[^9][^10][^8]

### 4.2 Improving Design Space Exploration (DSE)

Design space exploration for NPUs involves sweeping over array sizes, memory hierarchies, sparsity support, and parallelism strategies. SCALE‑Sim v3, for example, supports multi‑core simulations, sparsity, detailed DRAM modeling via Ramulator, and energy/power estimation via Accelergy, giving architects deep full‑system insights. NeuSim automates large‑scale exploration by parallelizing simulations across machines using Ray, enabling millions of hardware–software configurations to be evaluated.[^4][^7][^8][^9]

A TU cmodel contributes by ensuring that each design point evaluated in DSE is grounded in correct arithmetic semantics and realistic compute behavior. When combined with parametric timing models, the cmodel allows DSE frameworks to vary TU characteristics (e.g., pipeline depth, precision formats) while maintaining a consistent functional base.

### 4.3 Supporting Verification at Scale

As AI accelerators grow more complex, verification becomes a bottleneck. Functional C‑models enable high‑speed regression testing by running large suites of networks and micro‑benchmarks against both the cmodel and RTL representations. NPUsim’s full‑system, cycle‑accurate simulations demonstrate how such models can capture subtle bugs in scheduling, data hazards, and memory usage.[^10]

By integrating a TU cmodel via DPI‑C or similar interfaces into RTL testbenches, verification teams can check each TU operation’s result against the reference in real time, catching numerical mismatches, corner cases, and control flow anomalies early in the cycle. This dramatically reduces the cost of late‑stage bug fixes compared to discovering such issues in silicon or via FPGA emulation.

### 4.4 Integration with System Simulation and Emulation

High‑performance system simulators increasingly combine ISS (instruction set simulators), TLM (transaction‑level models), and hardware models to provide realistic software execution environments. Synopsys and others highlight the importance of cycle‑accurate hardware models in such systems for accurate performance estimation and driver development, but they also note the difficulty of hand‑coding these models, especially for custom accelerators.[^6]

A TU cmodel, possibly auto‑generated or partially derived from RTL, can be integrated into these system simulators as a peripheral device model. It can also serve as the compute engine behind FPGA prototypes or hybrid emulation setups, where the TU’s functional behavior runs in software while other components run in hardware, enabling fast iteration on TU design changes without rebuilding FPGA bitstreams.


## 5. Design Goals and Requirements for a TU C‑Model

### 5.1 Functional and Numerical Fidelity

The TU cmodel must faithfully implement the semantics of all supported instructions or micro‑ops:

- Matrix multiply‑accumulate operations (GEMM, convolution lowering, batched MM).
- Elementwise operations and fused epilogues (bias add, activation functions, normalization).[^16][^5]
- Data movement and tiling behavior between scratchpad/local SRAM and register files.[^5][^8]

Precision and numeric fidelity are critical: the model must match hardware rounding modes, accumulator widths, saturation behavior, underflow/overflow rules, NaN/Inf propagation, and quantization noise as specified by the architecture. This includes mixed‑precision accumulation (e.g., INT8 inputs with INT32 accumulators, BF16 inputs with FP32 accumulators) and any specialized formats (FP8, block‑floating‑point, etc.).[^1][^2]

### 5.2 Microarchitectural Abstraction and Configurability

The cmodel should expose a configurable parameter set describing the TU’s microarchitecture: array dimensions, pipeline depths, scratchpad sizes, bandwidths, supported dataflows, and sparsity formats. NPUsim’s configuration‑driven modeling of a wide variety of DNN accelerators (TPU, Eyeriss, Simba, etc.) demonstrates the value of such parameterization for reuse and research.[^7][^8][^10][^5]

This configurability allows a single TU cmodel implementation to serve many design points—useful both for internal architectural exploration and for sharing open‑source research artifacts without exposing proprietary microarchitectural details.

### 5.3 Performance and Scalability

A practical TU cmodel must be fast enough to run large workloads (full DNNs, multi‑batch inference traces) as part of daily regressions and DSE loops. NPU simulators like SCALE‑Sim, ONNXim, EONSim, NeuSim, and NPUsim emphasize efficient simulation, often by modeling compute deterministically and focusing detailed cycle‑level modeling on networks and memories.[^3][^4][^8][^9][^10][^7]

Following this example, a TU cmodel should:

- Use efficient data structures (e.g., blocking and tiling to exploit cache locality on the host CPU).
- Optionally leverage vectorized CPU instructions or GPU kernels for acceleration (careful to preserve exact numerical behavior).
- Support multi‑threading for parallel simulation of independent tiles or batch elements.

### 5.4 Interfaces and Integration Points

To serve as a reusable building block, the TU cmodel should provide:

- A C/C++ API that accepts tile descriptors, instruction encodings, and tensors, returning results and optional statistics.
- Bindings for higher‑level languages (Python, potentially Rust) for integration into research simulators and test frameworks.[^4][^8]
- DPI‑C or SystemC TLM interfaces for integration with RTL simulators (VCS, ModelSim, Verilator), enabling co‑simulation where TU RTL is compared against the cmodel for each operation.[^12][^1]

Integration patterns used in SAURIA (Python to generate stimuli and Verilator for RTL simulation) and NPUsim (ONNX/TensorFlow frontends feeding a configurable accelerator model) provide concrete precedents for how a TU cmodel can fit into a broader verification and research stack.[^16][^10]

### 5.5 Observability and Debuggability

Unlike physical hardware, a cmodel can expose internal state for debugging and analysis. Useful features include:

- Hooks to dump intermediate partial sums, tile buffers, and internal pipeline states for selected operations.
- Support for deterministic replay and logging, enabling reproducible bug reports.
- Built‑in self‑checking facilities (e.g., assertions on invariants such as accumulator ranges or memory alignment) to catch architectural contract violations early.

These capabilities are particularly valuable during microarchitecture bring‑up and when investigating discrepancies between cmodel predictions and RTL or silicon measurements.


## 6. Step‑by‑Step Implementation Roadmap

The following roadmap assumes a greenfield TU design within a modern AI accelerator project. It emphasizes staging to deliver early value while converging toward a robust, reusable cmodel.

### 6.1 Phase 0 – Requirements and Scope Definition

1. **Define TU instruction set and semantics**: Specify supported operations, data types, rounding rules, sparsity features, and fused operators. This specification should be unambiguous and testable.
2. **Decide target fidelity**: Start with a bit‑accurate functional model; plan for latency annotations or cycle‑approximate extensions if performance modeling is a near‑term goal.[^1][^3]
3. **Identify integration targets**: Determine which simulators (e.g., NPUsim‑like or custom), RTL environments, and compilers will consume the cmodel, and define the required APIs and throughput targets.

### 6.2 Phase 1 – Minimal Functional TU C‑Model

1. **Prototype core arithmetic kernels**: Implement the simplest TU operations (e.g., GEMM with a single data type, no sparsity) using clear, straightforward C++ with reference BLAS‑like loops.
2. **Establish test harness**: Build a unit test framework that compares cmodel outputs against trusted references (NumPy, cuBLAS, or framework kernels) for a wide range of tensor shapes and random seeds.
3. **Validate numerical edge cases**: Add tests covering max/min values, NaNs, Infs, denormals (if relevant), and quantization boundaries to lock down numerical behavior.[^2]

### 6.3 Phase 2 – Coverage of TU Feature Set

1. **Add dataflow and tiling semantics**: Model the TU’s dataflow policy (e.g., output‑stationary, weight‑stationary) and tile mapping between global and scratchpad memory.[^5][^16]
2. **Support multiple precisions and sparsity**: Implement INT8/BF16/FP16, mixed‑precision accumulation, and any structured sparsity modes, with corresponding tests.[^8][^7]
3. **Implement fused operations**: Add fused epilogues (bias, activation, normalization) that match the planned hardware implementation.
4. **Finalize instruction encoding support**: Implement a decoder that maps TU instruction encodings into internal operations, so compilers and assemblers can target the cmodel directly.

### 6.4 Phase 3 – Integration with RTL and System Simulators

1. **DPI‑C / SystemC interfaces**: Wrap the cmodel with DPI‑C interfaces so that SystemVerilog testbenches can call it as a golden reference during RTL simulation.[^12][^1]
2. **Co‑simulation environment**: Set up a co‑simulation flow where the same stimuli are fed to both TU RTL and cmodel, and outputs are compared cycle‑by‑cycle or transaction‑by‑transaction.
3. **Higher‑level simulator integration**: Integrate the cmodel into an NPU‑level simulator (internal or based on frameworks such as NPUsim/SCALE‑Sim‑style architecture), where TU calls correspond to tile‑level operations scheduled by the simulator.[^10][^8]

### 6.5 Phase 4 – Performance Modeling and DSE Support

1. **Latency and throughput modeling**: Introduce parametric latency estimators for TU operations, informed by microarchitectural analysis or simplified pipeline models. Follow ONNXim’s pattern of deterministic compute modeling per tile.[^15][^3]
2. **Energy and power estimates**: Integrate the cmodel with energy modeling tools (similar to EONSim’s integration of matrix compute models with memory energy) or frameworks like Accelergy, if available.[^7][^8]
3. **DSE tooling**: Expose knobs (array sizes, buffer depths, precision modes) and connect them to DSE frameworks that sweep hardware–software configurations, akin to NeuSim’s large‑scale exploration capabilities.[^9]

### 6.6 Phase 5 – Maintenance, Validation, and Release Strategy

1. **Regression infrastructure**: Maintain a continuous integration pipeline that runs the TU cmodel test suite and co‑simulation regressions on each change.
2. **Cross‑validation with silicon**: Once silicon is available, compare selected workloads’ outputs and performance against cmodel predictions, iteratively refining the model to reflect measured behavior.
3. **Documentation and potential open‑sourcing**: Document the cmodel’s interfaces, assumptions, and limitations. For academic or non‑proprietary designs, consider open‑sourcing a parameterized variant similar to NPUsim, SCALE‑Sim, or NeuSim to build a community and research impact.[^8][^9][^10]


## 7. Application Scenarios in Modern Chip Design

### 7.1 Pre‑Silicon Architecture and Algorithm Co‑Design

During the concept and feasibility phases, teams can use the TU cmodel embedded in an NPU simulator (internal or based on ideas from NPUsim, SCALE‑Sim, ONNXim, and EONSim) to test combinations of workloads, dataflows, and precision formats. This allows architects to answer questions like:[^3][^10][^7][^8]

- What array dimensions and buffer sizes balance utilization and memory bandwidth for targeted workloads?
- How do mixed‑precision configurations affect accuracy and throughput?
- Which sparsity formats yield meaningful benefits on real models?

Having a trusted TU cmodel ensures that differences in performance metrics are attributed to architectural changes rather than inconsistent functional behavior.

### 7.2 RTL Development and Verification

During RTL implementation, the TU cmodel becomes the reference against which each microarchitectural feature is validated. Testbenches can feed random and targeted vectors into the RTL and C‑model in lockstep, comparing outputs at operation boundaries. Functional coverage metrics can be defined in terms of TU opcodes, operand ranges, and dataflow patterns, with the cmodel serving as the oracle.

This mirrors established flows where MATLAB/Simulink or C‑based models are used to verify RTL implementations of DSP and control algorithms via cosimulation and DPI‑generated components. It extends those principles to specialized tensor compute units with AI workloads.[^12]

### 7.3 Software Stack Bring‑Up

The TU cmodel provides a stable target for compiler backends, schedulers, and runtime libraries. Compiler IR lowerings that generate TU instructions can be validated against the cmodel, and higher‑level frameworks (e.g., ONNX exporters) can be used to generate end‑to‑end workloads for simulation.

Simulators like ONNXim explicitly take ONNX models as input and map them onto multi‑core NPUs, using deterministic compute latency and detailed DRAM/NoC models. A TU cmodel integrated into such a system enables realistic end‑to‑end evaluation of the software stack long before hardware is available.[^15][^3]

### 7.4 Post‑Silicon Validation and Tuning

After first silicon, the cmodel aids in validating that the hardware implements the intended semantics. Discrepancies between silicon outputs and cmodel results may highlight undocumented behavior, implementation bugs, or model deficiencies. Performance gaps between cmodel‑based predictions and measured throughput guide further investigation into memory subsystems, interconnects, and control logic.

Tools like EONSim, which achieve low error in inference time and on‑chip memory access counts relative to real TPU hardware, illustrate how validated simulator models become long‑term assets for performance tuning and workload characterization. A TU cmodel is a crucial component of such validated simulators.[^7]


## 8. Risks, Limitations, and Mitigation Strategies

### 8.1 Modeling Drift and Maintenance Cost

A key risk is **modeling drift**—the divergence between the cmodel and evolving RTL or silicon behavior. As the TU microarchitecture and ISA evolve, the cmodel must be updated and retested. Without disciplined versioning and regression, the model may become obsolete or misleading.

Mitigation strategies include tightly coupling cmodel changes to specification changes, enforcing regression tests at each change, and using the cmodel only for behaviors explicitly covered by tests. Cross‑validation against NPUsim‑ or NeuSim‑like system simulators and eventual silicon helps catch drift early.[^9][^10]

### 8.2 Over‑Investment and Scope Creep

Another risk is over‑investing in modeling features that yield little incremental value, such as attempting full cycle accuracy for every internal signal when higher‑level timing annotations would suffice. Synopsys notes that creating C models for every custom RTL block can be impractical and requires specialized skills, which may not be justifiable in every project.[^6]

To avoid scope creep, teams should clearly define use cases and fidelity requirements up front, starting with the simplest model that meets verification and co‑development needs and only expanding scope when concrete benefits are identified.

### 8.3 Performance Bottlenecks in Large‑Scale Simulation

Poorly optimized cmodels can become bottlenecks in DSE and regression pipelines, particularly when simulating large networks or multi‑core systems. Tools like NeuSim and SCALE‑Sim emphasize scalable simulation through parallelization, efficient computation modeling, and focus on memory‑level contention.[^4][^8][^9]

By following similar strategies—vectorization, parallelism, and separation of deterministic compute from detailed memory modeling—a TU cmodel can remain performant enough for intensive usage.


## 9. Conclusion and Recommendations

Existing literature and open‑source projects in NPU simulation and C‑based verification strongly support the value of a TU cmodel for any serious AI accelerator project. C‑based reference models form a standard component of modern ASIC verification flows, particularly for algorithmic accelerators, where they serve as golden references, enable high‑speed functional verification, and support early software co‑development.[^2][^1][^12]

NPU simulators like NPUsim, SCALE‑Sim, NeuSim, ONNXim, and EONSim showcase the importance of accurate modeling of tensor compute and memory systems for architecture exploration, DSE, and full‑stack performance analysis. These frameworks often rely on deterministic or validated performance models of tensor computations, a role that a well‑designed TU cmodel can fill.[^10][^3][^8][^9][^7]

The research indicates that implementing at least a bit‑accurate functional TU cmodel is **highly recommended** for most modern AI accelerator projects, with extensions to latency‑annotated or cycle‑approximate models depending on performance modeling and DSE needs. Clear staging—from minimal functional core, through feature completion and RTL integration, to system‑level performance models—can deliver incremental value and manage complexity.

For a team building a new TU today, the recommended path is to:

1. Define a precise, testable TU ISA and numerical behavior specification.
2. Implement a minimal yet bit‑accurate TU cmodel focused on GEMM and convolution lowering.
3. Integrate this model into RTL verification via DPI‑C and into at least one NPU‑level simulator for architecture exploration.
4. Incrementally add timing and energy modeling informed by microarchitectural analysis and validated against hardware measurements.

If executed with discipline, a TU cmodel becomes a central artifact that aligns architects, RTL designers, verification engineers, and compiler/runtime teams, improving design quality and reducing risk across the chip’s lifecycle.

---

## References

1. [C-Based Verification in Hardware Design | PDF - Scribd](https://www.scribd.com/document/880636820/C-Based-Verification-in-VLSI-Design) - It is ideal for validating algorithms, creating reference models, and facilitating early design vali...

2. [Functional Simulation - an overview | ScienceDirect Topics](https://www.sciencedirect.com/topics/computer-science/functional-simulation) - Functional simulation refers to the process of testing and verifying the functionality of a computer...

3. [ONNXim: A Fast, Cycle-level Multi-core NPU Simulator - arXiv](https://arxiv.org/html/2406.08051v1)

4. [scalesim - PyPI](https://pypi.org/project/scalesim/) - SCALE Sim is a simulator for systolic array based accelerators for Convolution, Feed Forward, and an...

5. [Configurable Multi-directional Systolic Array Architecture for CNNs](https://dl.acm.org/doi/fullHtml/10.1145/3460776) - In this article, we design a configurable multi-directional systolic array (CMSA) to address these i...

6. [High-performance hardware models for system simulation](https://www.eenewseurope.com/en/high-performance-hardware-models-for-system-simulation/) - Chris Eddington of Synopsys focuses on how to use high-performance hardware models for system simula...

7. [EONSim: An NPU Simulator for On-Chip Memory and Embedding ...](https://arxiv.org/html/2511.06679)

8. [[Literature Review] SCALE-Sim v3: A modular cycle-accurate ...](https://www.themoonlight.io/en/review/scale-sim-v3-a-modular-cycle-accurate-systolic-accelerator-simulator-for-end-to-end-system-analysis) - The paper introduces SCALE-Sim v3, a modular, cycle-accurate simulator that enhances its predecessor...

9. [NeuSim Open-Source NPU Simulation Framework Released](https://www.linkedin.com/posts/jian-huang-16278625_github-platformxlabneusim-an-open-source-activity-7413316418898845696-9H-e) - We're pleased to release our open-source simulation framework NeuSim for NPU research. Check it at h...

10. [02-Group1-Bogil Kim.pptx](https://www.bnl.gov/modsim/events/2021/files/talks/bogil-kim.pdf)

11. [C based design methodology accelerates ASIC/FPGA design cycles](https://www.eetimes.com/c-based-design-methodology-accelerates-asic-fpga-design-cycles/) - C2R is a tool that allows designers to model their designs using C and then synthesize it into Veril...

12. [What Is ASIC Verification? - MATLAB & Simulink - MathWorks](https://www.mathworks.com/discovery/asic-verification.html) - ASIC verification ensures that a hardware implementation of an algorithm meets its specification. Ex...

13. [Tensor Processing Unit](https://en.wikipedia.org/wiki/Tensor_Processing_Unit) - Tensorflow, Jax, and PyTorch are supported frameworks for TPU. Google began using TPUs internally in...

14. [Architecture insights: MXU and TPU components](https://telnyx.com/learn-ai/mxu-tpu) - Architecture of TPU. A TPU chip contains multiple TensorCores, each consisting of an MXU, vector uni...

15. [[Literature Review] ONNXim: A Fast, Cycle-level Multi-core NPU ...](https://www.themoonlight.io/en/review/onnxim-a-fast-cycle-level-multi-core-npu-simulator) - The paper presents ONNXim, a fast, cycle-level simulator designed specifically for multi-core Neural...

16. [SAURIA (Systolic-Array tensor Unit for aRtificial ...](https://github.com/bsc-loca/sauria) - SAURIA (Systolic-Array tensor Unit for aRtificial Intelligence Acceleration) is an open-source Convo...

