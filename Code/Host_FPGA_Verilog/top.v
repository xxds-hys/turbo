
module top(
    input clk,
    input rst,
    input rd,
    input [8*128-1:0]   a0,
    input [8*128-1:0]   a1,
    input [31:0]        gamma0,
    input [31:0]        gamma1,
    input               start,

    output [8*128-1:0]  ao,
    output [31:0]       gammao
);

wire                isle;
wire [31:0]         gamma_bar;
wire [7:0]          w1;
wire [7:0]          w2;

wire [14:0]         sum_addr0;
wire [14:0]         sum_addr1;
wire [7:0]          sum0;
wire [7:0]          sum1;

reg [7:0]           cnt;
reg [8*128-1:0]     a0_reg;
reg [8*128-1:0]     a1_reg;
reg [7:0]           result;


assign sum_addr0 = isle? ({w1[6:0], result}) : ({w1[6:0], a1_reg[7:0]});
assign sum_addr1 = isle? ({w2[6:0], a1_reg[7:0]}) : ({w2[6:0], result});

abs_subtract u0(
    .gamma0     (gamma0),
    .gamma1     (gamma1),
    .isle       (isle),
    .gamma_bar  (gamma_bar)
);

rom256 u_rom256_w0(
    .clk    (clk),
    .en     (rst),
    .rd     (rd),
    .addr   (gamma_bar[7:0]),
    .dout   (w1)
);

rom256 u_rom256_w1(
    .clk    (clk),
    .en     (rst),
    .rd     (rd),
    .addr   (gamma_bar[7:0]),
    .dout   (w2)
);

rom256 u_rom256_gammao(
    .clk    (clk),
    .en     (rst),
    .rd     (rd),
    .addr   (gamma_bar[7:0]),
    .dout   (gammao)
);

rom32768 u_rom32768_w1(
    .clk    (clk),
    .en     (rst),
    .rd     (rd),
    .addr   (sum_addr0),
    .dout   (sum0)
);

rom32768 u_rom32768_w2(
    .clk    (clk),
    .en     (rst),
    .rd     (rd),
    .addr   (sum_addr1),
    .dout   (sum1)
);

always @(posedge clk) begin
    if(rst) begin
        a0_reg  <= 'h0;
        a1_reg  <= 'h0;
        result  <= 8'h0;
        cnt     <= 8'h8f;
    end
    else if(start) begin
        a0_reg  <= a0;
        a1_reg  <= a1;
        cnt     <= 8'h8f;
        result  <= a0[7:0];
    end
    else if(cnt != 8'h0) begin
        result <= sum0 + sum1;
        a0_reg <= (a0_reg >> 8);
        a1_reg <= (a1_reg >> 8);
        cnt <= cnt - 8'h1;
    end
end

assign ao = result;

endmodule

