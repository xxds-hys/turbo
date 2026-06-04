# NS-3 Simulation for Turbo

This directory contains the NS-3 simulation source code for Turbo, evaluating the system's performance with realistic workloads.

## Structure

*   **`src/`**: Modified NS-3 modules implementing Turbo's core logic.
    *   `internet/`: Modified IPv4/TCP stack for Deflow routing.
    *   `point-to-point/`: RDMA-capable network device (`qbb-net-device`) and switch logic.
    *   `network/`: Custom queue disciplines.
*   **`inputFiles/`**: Configuration and trace files.
    *   `TOPO.txt`: Network topology.
    *   `GPT_*.txt` / `LLAM_*.txt`: Workload traces for different LLM batch sizes.
*   **`Executable Files/`**: Simulation entry points and scripts.
    *   `A00082/main.cc`: Main simulation source file.
    *   `A00082/run.py`: Automation script for running experiments.

## Usage

1.  **Prerequisite**: Install NS-3 (version 3.33 recommended).
2.  **Setup**: Replace the default NS-3 `src/` directory with the `src/` directory provided in this repository to integrate Turbo's logic.
3.  **Build**: Recompile NS-3 (`./waf build`).
4.  **Run**:
    *   Copy `Executable Files/A00082/main.cc` to your NS-3 `scratch/` folder.
    *   Run using waf: `./waf --run "scratch/main --configFileName=INPUT_CONFIG --topoFileName=INPUT_TOPO ..."`
    *   Alternatively, `Executable Files/A00082/run.py` is provided to automate the setup and execution process.

