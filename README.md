# Turbo: Efficiently Serving Long-Context Large Language Models with In-Network Aggregation

This repository contains the source code and artifact scripts for the paper **"Turbo: Efficiently Serving Long-Context Large Language Models with In-Network Aggregation"**. 

Turbo is a system designed to accelerate long-context LLM serving by offloading attention calculation aggregation tasks to the network switch and programmable host NICs (FPGAs). This repository includes the P4 implementation for Intel Tofino 2 switches, Verilog code for FPGA-based host acceleration, and a packet-level NS-3 simulation environment.

## Repository Structure

```
.
├── Code/
│   ├── Host_FPGA_Verilog/       # Verilog source code for Host FPGA logic
│   ├── Simulator_NS3_CPP/       # NS-3 based simulation framework
│   │   ├── Executable Files/    # Scripts to run simulations
│   │   ├── inputFiles/          # Traffic traces (GPT, LLaMA) and topology configs
│   │   └── src/                 # Modified NS-3 source modules (routing, transport, etc.)
│   └── Switch_Tofino2_P4/       # P4_16 source code for Intel Tofino 2 switches
└── README.md
```

## Prerequisites

To fully reproduce the results and run the code, the following hardware and software environments are required:

### 1. In-Network Switch (P4)
*   **Hardware**: Intel Tofino 2 Programmable Switch.
*   **Software**: Intel P4 Studio (SDE) 9.7.0 or compatible version.

### 2. Host FPGA (Verilog)
*   **Software**: Standard FPGA synthesis tools (e.g., Intel Quartus Prime or Xilinx Vivado) compatible with your target FPGA platform.
*   **Hardware**: FPGA-equipped SmartNIC or development board.

### 3. Simulation (NS-3)
*   **OS**: Ubuntu 20.04/22.04 LTS (Recommended).
*   **Dependencies**: Python 3.x, C++ compiler (g++/clang), and standard NS-3 build dependencies.

## Getting Started

### Part 1: Switch Data Plane (P4)

The P4 code implements the in-network attention value aggregation.

1.  Navigate to the P4 directory:
    ```bash cd Code/Switch_Tofino2_P4/ ```
2.  The main P4 program is `attention_cal.p4`.
3.  Compile the program using the Tofino SDE compiler (`bf-p4c`). Ensure your environment variables for SDE are set.
    ```bash # Example compilation command (adjust according to your SDE path)
    bf-p4c -g --target tofino2 --arch t2na attention_cal.p4 ```
4.  Load the compiled binary onto the switch using the switch driver.

### Part 2: Host FPGA Logic (Verilog)

The Verilog code implements the host-side offloading logic.

1.  Navigate to the Verilog directory:
    ```bash cd Code/Host_FPGA_Verilog/ ```
2.  The top-level module is defined in `top.v`.
3.  The design includes basic arithmetic modules (e.g., `abs_subtract.v`) and ROMs for lookup tables.
4.  Import these files into your FPGA project workspace (Quartus/Vivado) for synthesis and implementation.

### Part 3: NS-3 Simulation

The simulation framework models the end-to-end performance of Turbo with traces from popular LLMs (GPT, LLaMA).

#### Setup
1.  Ensure you have `waf` and NS-3 dependencies installed.
2.  The simulation source code is located in `Code/Simulator_NS3_CPP/src/`.

#### Running Experiments
Execution scripts are provided in the `Executable Files` directory.

1.  Navigate to the execution directory:
    ```bash
    cd "Code/Simulator_NS3_CPP/Executable Files/A00082"
    ```
2.  Run the simulation using the provided Python script:
    ```bash python run.py```
    *   The script drives the simulation using input configurations found in `Code/Simulator_NS3_CPP/inputFiles/`.
    *   Traffic traces for different models (e.g., `GPT_1.txt`, `LLAM_16.txt`) define the workload patterns.

## Input Data

The `Code/Simulator_NS3_CPP/inputFiles/` directory contains configuration files used for evaluation:
*   **Traffic Patterns**: `GPT_*.txt`, `LLAM_*.txt` represent traffic traces for GPT and LLaMA models with varying batch sizes.
*   **Topology**: `TOPO.txt` defines the network topology.
*   **Protocol Config**: `CONFIG_DCQCN.txt` contains DCQCN parameters.
