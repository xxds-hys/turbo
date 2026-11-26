# Turbo: Efficiently Serving Long-Context Large Language Models with In-Network Aggregation

This repository contains the source code and artifact scripts for the paper **"Turbo: Efficiently Serving Long-Context Large Language Models with In-Network Aggregation"**, which has been accepted to SIGCOMM 2026.. 

Turbo is a system designed to accelerate long-context LLM serving by offloading attention calculation aggregation tasks to the network switch and programmable host NICs (FPGAs). This repository includes the P4 implementation for Intel Tofino 2 switches, Verilog code for FPGA-based host acceleration, and a packet-level NS-3 simulation environment.

## Repository Structure

```
.
├── Code/
│   ├── Host_FPGA_Verilog/           # Verilog source code for Host FPGA logic
│   ├── Simulator_NS3_CPP/           # NS-3.33 based simulation framework
│   │   ├── Dockerfile               # Container definition for build and precompiled runs
│   │   ├── README.md                # Detailed NS-3 simulation guide (English)
│   │   ├── README.zh-CN.md          # Detailed NS-3 simulation guide (Chinese)
│   │   ├── file-in-ctr/             # Experiment programs, input traces, and reference outputs
│   │   ├── ns-3.33/                 # Modified NS-3.33 simulator source code
│   │   └── scripts/                 # Unified build and run CLI (turbo.py)
│   └── Switch_Tofino2_P4/           # P4_16 source code for Intel Tofino 2 switches
│       ├── README.md                # Switch P4 compilation guide (English)
│       ├── README.zh-CN.md          # Switch P4 compilation guide (Chinese)
│       ├── attention_cal.p4         # Main P4 aggregation program
│       ├── headers.p4               # Packet headers and metadata definitions
│       └── parser.p4                # Parser and deparser definitions
└── README.md
```

## Prerequisites

To fully reproduce the results and run the code, the following hardware and software environments are required:

### 1. In-Network Switch (P4)
*   **Hardware**: Intel Tofino 2 Programmable Switch.
*   **Software**: Intel P4 Studio (SDE) (verified on 9.7.1 or compatible version).

### 2. Host FPGA (Verilog)
*   **Software**: Standard FPGA synthesis tools (e.g., Intel Quartus Prime or Xilinx Vivado) compatible with your target FPGA platform.
*   **Hardware**: FPGA-equipped SmartNIC or development board.

### 3. Simulation (NS-3)
*   **Option A — Docker (Recommended)**: Docker with Linux-container support (precompiled image published at `wzdl/turbo-ns3:v1.0`).
*   **Option B — Native Linux**: Linux x86_64 (verified on Ubuntu 20.04 amd64) with C/C++ build tools, Python 3 (`python3-dev`, `python3-distutils`), OpenMPI, Boost, GSL, SQLite3, and libxml2. See [Code/Simulator_NS3_CPP/README.md](Code/Simulator_NS3_CPP/README.md) for the exact package list.

## Getting Started

### Part 1: Switch Data Plane (P4)

The P4 code implements the in-network attention value aggregation. See [Code/Switch_Tofino2_P4/README.md](Code/Switch_Tofino2_P4/README.md) ([中文说明](Code/Switch_Tofino2_P4/README.zh-CN.md)) for full instructions.

1.  Navigate to the P4 directory:
    ```bash
    cd Code/Switch_Tofino2_P4/
    ```
2.  The main P4 program is `attention_cal.p4`.
3.  Compile the program using the Tofino SDE compiler (`bf-p4c`). Ensure your environment variables for SDE are set:
    ```bash
    bf-p4c --std p4-16 --target tofino2 --arch t2na -g -o ./build_out attention_cal.p4
    ```
4.  Load the compiled binary onto the switch using the switch driver.

### Part 2: Host FPGA Logic (Verilog)

The Verilog code implements the host-side offloading logic.

1.  Navigate to the Verilog directory:
    ```bash
    cd Code/Host_FPGA_Verilog/
    ```
2.  The top-level module is defined in `top.v`.
3.  The design includes basic arithmetic modules (e.g., `abs_subtract.v`) and ROMs for lookup tables.
4.  Import these files into your FPGA project workspace (Quartus/Vivado) for synthesis and implementation.

### Part 3: NS-3 Simulation

The simulation framework models the end-to-end performance of Turbo and baselines (`A00095`–`A00106`) with traces from popular LLMs (GPT, LLaMA). Full instructions, experiment catalog, and Windows PowerShell notes are provided in [Code/Simulator_NS3_CPP/README.md](Code/Simulator_NS3_CPP/README.md) ([中文说明](Code/Simulator_NS3_CPP/README.zh-CN.md)).

First, navigate to the NS-3 directory (except for Option 3, which does not require a local checkout):

```bash
cd Code/Simulator_NS3_CPP
```

Three ways are supported to build and run the experiments (using `A00095` as an example; replace with `all` to run all 12 experiments):

1.  **Build and run manually (Linux)**:
    ```bash
    python3 scripts/turbo.py build --configure --jobs 2
    python3 scripts/turbo.py run A00095
    ```
2.  **Compile and run in Docker** (mount local source into the container toolchain):
    ```bash
    docker pull "wzdl/turbo-ns3:v1.0"
    docker run --rm --mount "type=bind,source=$(pwd),target=/opt/turbo" "wzdl/turbo-ns3:v1.0" build --configure --jobs 2
    docker run --rm --mount "type=bind,source=$(pwd),target=/opt/turbo" "wzdl/turbo-ns3:v1.0" run A00095
    ```
3.  **Run the precompiled Docker image directly** (no compilation required):
    ```bash
    docker pull "wzdl/turbo-ns3:v1.0"
    mkdir -p results
    docker run --rm --mount "type=bind,source=$(pwd)/results,target=/results" "wzdl/turbo-ns3:v1.0" run A00095 --output-dir /results
    ```

## Input Data

The `Code/Simulator_NS3_CPP/file-in-ctr/inputFiles/` directory contains per-experiment (`A00095`–`A00106`) configuration files used for evaluation:
*   **Traffic Patterns**: `GPT_*.txt`, `LLAM_*.txt` represent traffic traces for GPT and LLaMA models with varying batch sizes (`1`, `4`, `8`, `16`, `32`).
*   **Topology**: `TOPO.txt` defines the network topology (Spark or Spine-Leaf, 200 Gbps links).
*   **Protocol Config**: `CONFIG_DCQCN.txt` contains DCQCN parameters.

Reference outputs for all experiments are stored in `Code/Simulator_NS3_CPP/file-in-ctr/outputFiles/`.
