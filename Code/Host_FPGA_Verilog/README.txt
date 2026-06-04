# Host FPGA Verilog Code

This directory contains the Verilog source code for the Turbo host-side acceleration logic.

## Files
*   `top.v`: Top-level module interface.
*   `abs_subtract.v`: Arithmetic module for calculating absolute differences.
*   `rom*.v`: ROM modules for lookup tables used in the computation.

## Usage
These files are intended to be synthesized for an FPGA target (e.g., on a SmartNIC).
1.  Create a new project in your synthesis tool (Quartus/Vivado).
2.  Add `top.v` and dependent modules.
3.  Assign pins according to your specific board constraints.
4.  Synthesize and generate the bitstream.

