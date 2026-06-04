/*
module rom32768 (
    input [14:0] addr,
    output [7:0] data
);

reg [7:0] memory [0:32767];
integer i;

initial begin
    for (i = 0; i < 32767; i = i + 1) begin
        memory[i] = i;
    end
end

assign data = memory[addr];

endmodule
*/

module rom32768 (
    input clk,
    input en,
    input rd,
    input [14:0] addr,
    output reg [7:0] dout
);

(*rom_style = "block" *) reg [7:0] data [0:32767];

always @(posedge clk) begin
    if (en)
        data[addr] <= addr;
    //case(addr)
    //    8'b00000000: data <= 8'h0A; 8'b00100000: data <= 8'h22; 8'b01100000: data <= 8'h22; 8'b11100000: data <= 8'h22;
    //    8'b00000001: data <= 8'h00; 8'b00100001: data <= 8'h01; 8'b01100001: data <= 8'h01; 8'b11100001: data <= 8'h01;
    //    8'b00000010: data <= 8'h01; 8'b00100010: data <= 8'h42; 8'b01100010: data <= 8'h42; 8'b11100010: data <= 8'h42;
    //    8'b00000011: data <= 8'h00; 8'b00100011: data <= 8'h2B; 8'b01100011: data <= 8'h2B; 8'b11100011: data <= 8'h2B;
    //    8'b00000100: data <= 8'h01; 8'b00100100: data <= 8'h00; 8'b01100100: data <= 8'h00; 8'b11100100: data <= 8'h00;
    //    8'b00000101: data <= 8'h3A; 8'b00100101: data <= 8'h02; 8'b01100101: data <= 8'h02; 8'b11100101: data <= 8'h02;
    //    8'b00000110: data <= 8'h00; 8'b00100110: data <= 8'h02; 8'b01100110: data <= 8'h02; 8'b11100110: data <= 8'h02;
    //    8'b00000111: data <= 8'h02; 8'b00100111: data <= 8'h02; 8'b01100111: data <= 8'h02; 8'b11100111: data <= 8'h02;
    //    8'b00001000: data <= 8'h10; 8'b00101000: data <= 8'h00; 8'b01101000: data <= 8'h00; 8'b11101000: data <= 8'h00;
    //    8'b00001001: data <= 8'h3B; 8'b00101001: data <= 8'h01; 8'b01101001: data <= 8'h01; 8'b11101001: data <= 8'h01;
    //    8'b00001010: data <= 8'h00; 8'b00101010: data <= 8'h23; 8'b01101010: data <= 8'h23; 8'b11101010: data <= 8'h23;
    //    8'b00001011: data <= 8'h02; 8'b00101011: data <= 8'h03; 8'b01101011: data <= 8'h03; 8'b11101011: data <= 8'h03;
    //    8'b00001100: data <= 8'h01; 8'b00101100: data <= 8'h33; 8'b01101100: data <= 8'h33; 8'b11101100: data <= 8'h33;
    //    8'b00001101: data <= 8'h00; 8'b00101101: data <= 8'h01; 8'b01101101: data <= 8'h01; 8'b11101101: data <= 8'h01;
    //    8'b00001110: data <= 8'h01; 8'b00101110: data <= 8'h04; 8'b01101110: data <= 8'h04; 8'b11101110: data <= 8'h04;
    //    8'b00001111: data <= 8'h00; 8'b00101111: data <= 8'h01; 8'b01101111: data <= 8'h01; 8'b11101111: data <= 8'h01;
    //    8'b00010000: data <= 8'h40; 8'b00110000: data <= 8'h02; 8'b01110000: data <= 8'h02; 8'b11110000: data <= 8'h02;
    //    8'b00010001: data <= 8'h41; 8'b00110001: data <= 8'h37; 8'b01110001: data <= 8'h37; 8'b11110001: data <= 8'h37;
    //    8'b00010010: data <= 8'h02; 8'b00110010: data <= 8'h36; 8'b01110010: data <= 8'h36; 8'b11110010: data <= 8'h36;
    //    8'b00010011: data <= 8'h00; 8'b00110011: data <= 8'h01; 8'b01110011: data <= 8'h01; 8'b11110011: data <= 8'h01;
    //    8'b00010100: data <= 8'h01; 8'b00110100: data <= 8'h02; 8'b01110100: data <= 8'h02; 8'b11110100: data <= 8'h02;
    //    8'b00010101: data <= 8'h00; 8'b00110101: data <= 8'h37; 8'b01110101: data <= 8'h37; 8'b11110101: data <= 8'h37;
    //    8'b00010110: data <= 8'h01; 8'b00110110: data <= 8'h04; 8'b01110110: data <= 8'h04; 8'b11110110: data <= 8'h04;
    //    8'b00010111: data <= 8'h02; 8'b00110111: data <= 8'h04; 8'b01110111: data <= 8'h04; 8'b11110111: data <= 8'h04;
    //    8'b00011000: data <= 8'h03; 8'b00111000: data <= 8'h40; 8'b01111000: data <= 8'h40; 8'b11111000: data <= 8'h40;
    //    8'b00011001: data <= 8'h1E; 8'b00111001: data <= 8'h00; 8'b01111001: data <= 8'h00; 8'b11111001: data <= 8'h00;
    //    8'b00011010: data <= 8'h01; 8'b00111010: data <= 8'h00; 8'b01111010: data <= 8'h00; 8'b11111010: data <= 8'h00;
    //    8'b00011011: data <= 8'h02; 8'b00111011: data <= 8'h00; 8'b01111011: data <= 8'h00; 8'b11111011: data <= 8'h00;
    //    8'b00011100: data <= 8'h22; 8'b00111100: data <= 8'h0D; 8'b01111100: data <= 8'h0D; 8'b11111100: data <= 8'h0D;
    //    8'b00011101: data <= 8'h21; 8'b00111101: data <= 8'h41; 8'b01111101: data <= 8'h41; 8'b11111101: data <= 8'h41;
    //    8'b00011110: data <= 8'h01; 8'b00111110: data <= 8'h01; 8'b01111110: data <= 8'h01; 8'b11111110: data <= 8'h01;
    //    8'b00011111: data <= 8'h02; 8'b00111111: data <= 8'h0D; 8'b01111111: data <= 8'h0D; 8'b11111111: data <= 8'h0D;
    //endcase
end

always @(posedge clk)
begin
    if(rd)
        dout<=data[addr];    
end


endmodule
