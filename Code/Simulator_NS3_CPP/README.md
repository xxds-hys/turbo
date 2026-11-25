# Turbo ns-3 Simulation Guide

[中文说明](README.zh-CN.md)

Turbo's simulator is built on ns-3.33. This repository contains
the simulator source, experiment programs, configurations, workload traces,
and reference outputs.

## Repository Layout

| Directory | Contents |
| --- | --- |
| `ns-3.33/` | Simulator source |
| `file-in-ctr/executableFiles/` | Experiment programs |
| `file-in-ctr/inputFiles/` | Configurations, topologies, and workload traces |
| `file-in-ctr/outputFiles/` | Reference outputs |
| `scripts/` | Build and run entry points |

Run source-related commands from this directory (`Code/Simulator_NS3_CPP/`). Shell examples below use
Bash. On Windows, we recommend WSL2 with Docker Desktop's Linux-container and
WSL2 integration enabled, or running the Docker commands directly in PowerShell
(see the note in Section 3).

## 1. Build and Run Manually

### Environment and Dependencies

- **Platform**: Linux x86_64 (verified on Ubuntu 20.04 amd64)
- **Required packages**:
  - C/C++ build tools: `build-essential`, `pkg-config`
  - Python 3 environment: `python3`, `python3-dev`, `python3-distutils`
  - MPI and libraries: `openmpi-bin`, `libopenmpi-dev`, `libboost-all-dev`, `libgsl-dev`, `libsqlite3-dev`, `libxml2-dev`

Please install the dependencies above for your system (reference command for Ubuntu 20.04):

```bash
sudo apt-get update && sudo apt-get install -y build-essential python3 python3-dev python3-distutils pkg-config openmpi-bin libopenmpi-dev libboost-all-dev libgsl-dev libsqlite3-dev libxml2-dev
```

### Build and Run

Once the environment is ready, build the simulator and run `A00095`:

```bash
python3 scripts/turbo.py build --configure --jobs 2
python3 scripts/turbo.py run A00095
```

Results are written to `artifacts/results/A00095/`.

## 2. Compile and Run in Docker

Install Docker with Linux-container support. The image includes the source,
build toolchain, and precompiled experiments.

```bash
docker pull "wzdl/turbo-ns3:v1.0"
```

Mount this repository to compile your local source inside the container environment:

```bash
docker run --rm --mount "type=bind,source=$(pwd),target=/opt/turbo" "wzdl/turbo-ns3:v1.0" build --configure --jobs 2
docker run --rm --mount "type=bind,source=$(pwd),target=/opt/turbo" "wzdl/turbo-ns3:v1.0" run A00095
```

Results are saved to `artifacts/results/A00095/` on the host. After modifying
source files, rerun `build` before running the experiment.

## 3. Run the Precompiled Docker Image

This option requires only Docker and the published image (no checkout or compilation needed):

```bash
docker pull "wzdl/turbo-ns3:v1.0"
mkdir -p results
docker run --rm --mount "type=bind,source=$(pwd)/results,target=/results" "wzdl/turbo-ns3:v1.0" run A00095 --output-dir /results
```

> **Windows note**: When running directly in Windows PowerShell (instead of WSL2/Bash), replace `mkdir -p results` with `New-Item -ItemType Directory -Force results`. The `docker run` commands work as-is.

Results are saved to `results/A00095/` on the host.

## Run Other Experiments

Replace `A00095` with the experiment name in the commands above. To run all
experiments:

```bash
python3 scripts/turbo.py run all
```

Using the precompiled image:

```bash
docker run --rm --mount "type=bind,source=$(pwd)/results,target=/results" "wzdl/turbo-ns3:v1.0" run all --output-dir /results
```

To select one workload:

```bash
python3 scripts/turbo.py run A00095 --workload GPT_1
```

Workload names are `GPT_1`, `GPT_4`, `GPT_8`, `GPT_16`, `GPT_32`, and the
corresponding `LLAM_*` names. Output files include `*-FCT.txt` for completion
records and, when queue monitoring is enabled, `*-QLEN.txt` for queue samples.
Pass a different `--output-dir` to retain results across repeated runs.

## Experiment Catalog

All experiments below use 200 Gbps links.

| Experiment | Method | Topology | Workloads |
| --- | --- | --- | --- |
| A00095 | Turbo | Spark | Ten GPT/LLaMA traces |
| A00096 | Turbo | Spine-Leaf | Ten GPT/LLaMA traces |
| A00097 | Pass-KV (Ring) | Spine-Leaf | Ten GPT/LLaMA traces |
| A00098 | Pass-KV (Ring) | Spark | Ten GPT/LLaMA traces |
| A00099 | Pass-KV (P2P) | Spark | Ten GPT/LLaMA traces |
| A00100 | Pass-KV (P2P) | Spine-Leaf | Ten GPT/LLaMA traces |
| A00101 | Pass-Q (P2P) | Spine-Leaf | Ten GPT/LLaMA traces |
| A00102 | Pass-Q (P2P) | Spark | Ten GPT/LLaMA traces |
| A00103 | Pass-Q (Ring) | Spark | Ten GPT/LLaMA traces |
| A00104 | Pass-Q (Ring) | Spine-Leaf | Ten GPT/LLaMA traces |
| A00105 | Pass-KV (P2P), queue monitoring | Spark | GPT-32 |
| A00106 | Pass-Q (P2P), queue monitoring | Spark | GPT-32 |
