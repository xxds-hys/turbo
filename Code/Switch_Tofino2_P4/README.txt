# Switch P4 Code

This directory contains the P4_16 source code for the Turbo switch data plane, designed for the Intel Tofino 2 platform.

## Files
*   `attention_cal.p4`: Main P4 program logic for attention value aggregation.
*   `headers.p4`: Header definitions.
*   `parser.p4`: Parser logic.
*   `build/`: Build artifacts and makefiles.

## Compilation
Use the Intel P4 Studio (SDE) compiler:
```bash
bf-p4c -g --target tofino2 --arch t2na attention_cal.p4
```

