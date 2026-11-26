# Turbo Tofino 2 Switch P4 Guide

[中文说明](README.zh-CN.md)

This directory contains the `P4_16` data-plane source code for Turbo's in-network
attention aggregation, targeting Intel Tofino 2 programmable switches (`t2na`).

## File Layout

| File | Contents |
| --- | --- |
| `attention_cal.p4` | Main P4 program (Ingress/Egress controls, lookup tables, registers, and rolling-forward pipeline) |
| `headers.p4` | Custom packet headers, metadata structures, and macro definitions |
| `parser.p4` | Ingress/Egress parser and deparser definitions |
| `manifest.json` | Build manifest |

Run all commands below from this directory (`Code/Switch_Tofino2_P4/`).

## Prerequisites

- **Hardware**: Intel Tofino 2 programmable switch
- **Software**: Intel P4 Studio SDE (verified with `bf-sde-9.7.1`; compatible 9.x versions supported)

## Compilation

1. Set the Intel P4 Studio SDE environment variables (adjust `/path/to/bf-sde` to your installation path):

```bash
export SDE=/path/to/bf-sde
export SDE_INSTALL=$SDE/install
export PATH=$SDE_INSTALL/bin:$PATH
```

2. Compile `attention_cal.p4` with `bf-p4c`:

```bash
bf-p4c --std p4-16 --target tofino2 --arch t2na -g -o ./build_out attention_cal.p4
```

Compiled artifacts and compiler logs are written to `./build_out/` (for example,
`./build_out/manifest.json` and `./build_out/pipe/logs/mau.resources.log`).

If your SDE setup uses `p4_build.sh`, you can also build with:

```bash
$SDE/p4_build.sh attention_cal.p4 --with-tofino2
```

3. Load the compiled program onto the switch using the Intel Tofino switch driver (`bf_switchd`) and populate the runtime lookup tables and forwarding rules from the control plane.
