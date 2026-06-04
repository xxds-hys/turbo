
module abs_subtract(
    input [31:0]    gamma0,
    input [31:0]    gamma1,
    output          isle, //1: gamma0 < gamma1
    output [31:0]   gamma_bar
);

wire [31:0] sub_result;

assign sub_result = gamma0 - gamma1;
assign isle = sub_result[31];

assign gamma_bar = isle ? (~sub_result + 1) : sub_result;

endmodule

