# Turbo Tofino 2 交换机 P4 编译指南

[English](README.md)

本目录包含 Turbo 网内注意力聚合数据面的 `P4_16` 源码，面向 Intel Tofino 2 可编程交换机（`t2na` 架构）。

## 文件说明

| 文件 | 内容 |
| --- | --- |
| `attention_cal.p4` | 主 P4 程序（Ingress/Egress 控制逻辑、查找表、寄存器及前滚更新流水线） |
| `headers.p4` | 自定义协议头、元数据结构体及宏定义 |
| `parser.p4` | Ingress/Egress 解析器与逆解析器定义 |

以下命令均在本目录（`Code/Switch_Tofino2_P4/`）下执行。

## 环境要求

- **硬件**：Intel Tofino 2 可编程交换机
- **软件**：Intel P4 Studio SDE（已在 `bf-sde-9.7.1` 上验证，兼容同系列 9.x 版本）

## 编译方法

1. 配置 Intel P4 Studio SDE 环境变量（请将 `/path/to/bf-sde` 替换为实际安装路径）：

```bash
export SDE=/path/to/bf-sde
export SDE_INSTALL=$SDE/install
export PATH=$SDE_INSTALL/bin:$PATH
```

2. 使用 `bf-p4c` 编译 `attention_cal.p4`：

```bash
bf-p4c --std p4-16 --target tofino2 --arch t2na -g -o ./build_out attention_cal.p4
```

编译产物与日志将输出到 `./build_out/` 目录（例如 `./build_out/manifest.json` 与 `./build_out/pipe/logs/mau.resources.log`）。

如果当前 SDE 环境习惯使用 `p4_build.sh`，也可以通过以下命令编译：

```bash
$SDE/p4_build.sh attention_cal.p4 --with-tofino2
```

3. 编译完成后，通过 Intel Tofino 交换机驱动（`bf_switchd`）加载编译产物，并由控制面下发查找表项与转发规则。
