# Turbo ns-3 仿真指南

[English](README.md)

本项目基于 ns-3.33，包含模拟器源码、实验程序、配置文件、工作负载轨迹及参考输出。

## 目录结构

| 目录 | 内容 |
| --- | --- |
| `ns-3.33/` | 模拟器源码 |
| `file-in-ctr/executableFiles/` | 实验程序 |
| `file-in-ctr/inputFiles/` | 配置、拓扑和工作负载轨迹 |
| `file-in-ctr/outputFiles/` | 参考输出结果 |
| `scripts/` | 编译和运行入口 |

涉及源码的命令均在本目录（`Code/Simulator_NS3_CPP/`）执行。以下示例默认使用 Bash；Windows 用户建议在 WSL2 中运行（并启用 Docker Desktop 的 Linux 容器及 WSL2 集成），或直接在 PowerShell 中运行 Docker 命令（见方式 3 提示）。

## 1. 自行准备环境并从源码编译运行

### 环境与依赖

- **系统环境**：Linux x86_64（已在 Ubuntu 20.04 amd64 上验证）
- **编译与运行依赖**：
  - C/C++ 编译工具链：`build-essential`、`pkg-config`
  - Python 3 环境：`python3`、`python3-dev`、`python3-distutils`
  - MPI 与基础库：`openmpi-bin`、`libopenmpi-dev`、`libboost-all-dev`、`libgsl-dev`、`libsqlite3-dev`、`libxml2-dev`

请根据自身系统提前安装上述依赖（以下提供 Ubuntu 20.04 的参考安装命令）：

```bash
sudo apt-get update && sudo apt-get install -y build-essential python3 python3-dev python3-distutils pkg-config openmpi-bin libopenmpi-dev libboost-all-dev libgsl-dev libsqlite3-dev libxml2-dev
```

### 编译并运行

环境就绪后，编译模拟器并运行 `A00095`：

```bash
python3 scripts/turbo.py build --configure --jobs 2
python3 scripts/turbo.py run A00095
```

结果保存在 `artifacts/results/A00095/`。

## 2. 使用 Docker 环境编译运行

安装支持 Linux 容器的 Docker。镜像包含完整源码、编译环境与预编译实验程序。

```bash
docker pull "wzdl/turbo-ns3:v1.0"
```

挂载当前仓库，使用镜像中的环境编译本机源码：

```bash
docker run --rm --mount "type=bind,source=$(pwd),target=/opt/turbo" "wzdl/turbo-ns3:v1.0" build --configure --jobs 2
docker run --rm --mount "type=bind,source=$(pwd),target=/opt/turbo" "wzdl/turbo-ns3:v1.0" run A00095
```

结果保存在本机仓库的 `artifacts/results/A00095/`。修改源码后，重新执行 `build` 再运行实验。

## 3. 直接运行 Docker 预编译镜像

此方式无需下载源码或编译，只需要 Docker 和已发布的镜像：

```bash
docker pull "wzdl/turbo-ns3:v1.0"
mkdir -p results
docker run --rm --mount "type=bind,source=$(pwd)/results,target=/results" "wzdl/turbo-ns3:v1.0" run A00095 --output-dir /results
```

> **Windows 提示**：若直接在 Windows PowerShell（而非 WSL2/Bash）中执行上述命令，请将 `mkdir -p results` 替换为 `New-Item -ItemType Directory -Force results`；两条 `docker run` 命令可直接原样运行。

结果保存在宿主机的 `results/A00095/`。

## 运行其他实验

将上述命令中的 `A00095` 替换为对应实验名称。运行全部实验：

```bash
python3 scripts/turbo.py run all
```

使用预编译镜像运行全部实验：

```bash
docker run --rm --mount "type=bind,source=$(pwd)/results,target=/results" "wzdl/turbo-ns3:v1.0" run all --output-dir /results
```

仅运行一种工作负载：

```bash
python3 scripts/turbo.py run A00095 --workload GPT_1
```

工作负载名称为 `GPT_1`、`GPT_4`、`GPT_8`、`GPT_16`、`GPT_32`，以及对应的 `LLAM_*` 名称。`*-FCT.txt` 保存完成事件记录，开启队列监测时还会生成 `*-QLEN.txt`。需要保留多次运行结果时，请指定不同的 `--output-dir`。

## 实验目录

以下实验均使用 200 Gbps 链路。

| 实验 | 方法 | 拓扑 | 工作负载 |
| --- | --- | --- | --- |
| A00095 | Turbo | Spark | 十种 GPT/LLaMA 轨迹 |
| A00096 | Turbo | Spine-Leaf | 十种 GPT/LLaMA 轨迹 |
| A00097 | Pass-KV (Ring) | Spine-Leaf | 十种 GPT/LLaMA 轨迹 |
| A00098 | Pass-KV (Ring) | Spark | 十种 GPT/LLaMA 轨迹 |
| A00099 | Pass-KV (P2P) | Spark | 十种 GPT/LLaMA 轨迹 |
| A00100 | Pass-KV (P2P) | Spine-Leaf | 十种 GPT/LLaMA 轨迹 |
| A00101 | Pass-Q (P2P) | Spine-Leaf | 十种 GPT/LLaMA 轨迹 |
| A00102 | Pass-Q (P2P) | Spark | 十种 GPT/LLaMA 轨迹 |
| A00103 | Pass-Q (Ring) | Spark | 十种 GPT/LLaMA 轨迹 |
| A00104 | Pass-Q (Ring) | Spine-Leaf | 十种 GPT/LLaMA 轨迹 |
| A00105 | Pass-KV (P2P)，队列监测 | Spark | GPT-32 |
| A00106 | Pass-Q (P2P)，队列监测 | Spark | GPT-32 |
