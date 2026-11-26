#define ATTENTION_DIM 128

#define ATTENTION_WIDTH 8
#define GAMMA_WIDTH 4

// Per paper §5.1 & §6.3: 11-bit lookup table (2^11 = 2048 entries, 2 KB)
#define ATTENTION_CAL_TABLE_SIZE 2048

// 14B
header ethernet_h {
    bit<48> dst_addr;
    bit<48> src_addr;
    bit<16> ether_type;
}

// 20B
header ipv4_h {
    bit<4>   version;
    bit<4>   ihl;
    bit<8>   diffserv;
    bit<16>  total_len;
    bit<16>  identification;
    bit<3>   flags;
    bit<13>  frag_offset;
    bit<8>   ttl;
    bit<8>   protocol;
    bit<16>  hdr_checksum;
    bit<32>  src_addr;
    bit<32>  dst_addr;
}

// 8B
header udp_h {
    bit<16>    src_port;
    bit<16>    dst_port;
    bit<16>    len;
    bit<16>    checksum;
}

// 4B
header attention_h {
    bit<8> layer_id;
    bit<8> head_id;
    bit<8> seq_id;
    bit<8> gamma;
}

// 2B
header recirculate_h {
    bit<8> recirculate_flag; // 220 is recirculate pkt
    bit<8> recirculate_tag;  // 0: gamma; 1: attention and m, n
}

header recirculate_gamma_h {
    bit<8> gamma_diff;
    bit<8> gamma_local;
    bit<8> gamma_new;
}

header recirculate_attention_h {
    bit<8> m; // 1/(1+2^gamma)
    bit<8> n; // 1 - 1/(1+2^gamma)
    bit<8> data_rec_0;
    bit<8> data_rec_1;
    bit<8> data_rec_2;
    bit<8> data_rec_3;
    bit<8> data_rec_4;
    bit<8> data_rec_5;
    bit<8> data_rec_6;
    bit<8> data_rec_7;
    bit<8> data_rec_8;
    bit<8> data_rec_9;
    bit<8> data_rec_10;
    bit<8> data_rec_11;
    bit<8> data_rec_12;
    bit<8> data_rec_13;
    bit<8> data_rec_14;
    bit<8> data_rec_15;
}

// 32-element segment (paper §5.1: 32-element configuration, 8 stages, plateau = 6)
header attention_data_h {
    bit<ATTENTION_WIDTH> data_0;
    bit<ATTENTION_WIDTH> data_1;
    bit<ATTENTION_WIDTH> data_2;
    bit<ATTENTION_WIDTH> data_3;
    bit<ATTENTION_WIDTH> data_4;
    bit<ATTENTION_WIDTH> data_5;
    bit<ATTENTION_WIDTH> data_6;
    bit<ATTENTION_WIDTH> data_7;
    bit<ATTENTION_WIDTH> data_8;
    bit<ATTENTION_WIDTH> data_9;
    bit<ATTENTION_WIDTH> data_10;
    bit<ATTENTION_WIDTH> data_11;
    bit<ATTENTION_WIDTH> data_12;
    bit<ATTENTION_WIDTH> data_13;
    bit<ATTENTION_WIDTH> data_14;
    bit<ATTENTION_WIDTH> data_15;
    bit<ATTENTION_WIDTH> data_16;
    bit<ATTENTION_WIDTH> data_17;
    bit<ATTENTION_WIDTH> data_18;
    bit<ATTENTION_WIDTH> data_19;
    bit<ATTENTION_WIDTH> data_20;
    bit<ATTENTION_WIDTH> data_21;
    bit<ATTENTION_WIDTH> data_22;
    bit<ATTENTION_WIDTH> data_23;
    bit<ATTENTION_WIDTH> data_24;
    bit<ATTENTION_WIDTH> data_25;
    bit<ATTENTION_WIDTH> data_26;
    bit<ATTENTION_WIDTH> data_27;
    bit<ATTENTION_WIDTH> data_28;
    bit<ATTENTION_WIDTH> data_29;
    bit<ATTENTION_WIDTH> data_30;
    bit<ATTENTION_WIDTH> data_31;
}

struct ingress_header_t {
    recirculate_h recirculate;
    recirculate_gamma_h recirculate_gamma;
    recirculate_attention_h recirculate_attention;
    ethernet_h ethernet;
    ipv4_h ipv4;
    udp_h udp;
    attention_h attention;
    attention_data_h attention_data;
}

struct egress_header_t {

}

struct egress_metadata_t {

}

struct attention_set_t {
    bit<ATTENTION_WIDTH> data_0;
    bit<ATTENTION_WIDTH> data_1;
    bit<ATTENTION_WIDTH> data_2;
    bit<ATTENTION_WIDTH> data_3;
    bit<ATTENTION_WIDTH> data_4;
    bit<ATTENTION_WIDTH> data_5;
    bit<ATTENTION_WIDTH> data_6;
    bit<ATTENTION_WIDTH> data_7;
    bit<ATTENTION_WIDTH> data_8;
    bit<ATTENTION_WIDTH> data_9;
    bit<ATTENTION_WIDTH> data_10;
    bit<ATTENTION_WIDTH> data_11;
    bit<ATTENTION_WIDTH> data_12;
    bit<ATTENTION_WIDTH> data_13;
    bit<ATTENTION_WIDTH> data_14;
    bit<ATTENTION_WIDTH> data_15;
    bit<ATTENTION_WIDTH> data_16;
    bit<ATTENTION_WIDTH> data_17;
    bit<ATTENTION_WIDTH> data_18;
    bit<ATTENTION_WIDTH> data_19;
    bit<ATTENTION_WIDTH> data_20;
    bit<ATTENTION_WIDTH> data_21;
    bit<ATTENTION_WIDTH> data_22;
    bit<ATTENTION_WIDTH> data_23;
    bit<ATTENTION_WIDTH> data_24;
    bit<ATTENTION_WIDTH> data_25;
    bit<ATTENTION_WIDTH> data_26;
    bit<ATTENTION_WIDTH> data_27;
    bit<ATTENTION_WIDTH> data_28;
    bit<ATTENTION_WIDTH> data_29;
    bit<ATTENTION_WIDTH> data_30;
    bit<ATTENTION_WIDTH> data_31;
}

struct ingress_metadata_t {
    bit<8> offset;
    bit<8> gamma_ba;
    bit<8> gamma_log_res;
    bit<8> gamma_max;
    bit<4> gamma_ba_cal_res;     // 4-bit quantized delta_gamma + 1-bit flag = 5 bits (paper m1=5)
    bit<1> local_gamma_big_flag;
    attention_set_t attention_set;

    #define ATTENTION_RES(i) \
        bit<ATTENTION_WIDTH> attention_##i##_pkt_res; \
        bit<ATTENTION_WIDTH> attention_##i##_local_res;

    ATTENTION_RES(0)
    ATTENTION_RES(1)
    ATTENTION_RES(2)
    ATTENTION_RES(3)
    ATTENTION_RES(4)
    ATTENTION_RES(5)
    ATTENTION_RES(6)
    ATTENTION_RES(7)
    ATTENTION_RES(8)
    ATTENTION_RES(9)
    ATTENTION_RES(10)
    ATTENTION_RES(11)
    ATTENTION_RES(12)
    ATTENTION_RES(13)
    ATTENTION_RES(14)
    ATTENTION_RES(15)
    ATTENTION_RES(16)
    ATTENTION_RES(17)
    ATTENTION_RES(18)
    ATTENTION_RES(19)
    ATTENTION_RES(20)
    ATTENTION_RES(21)
    ATTENTION_RES(22)
    ATTENTION_RES(23)
    ATTENTION_RES(24)
    ATTENTION_RES(25)
    ATTENTION_RES(26)
    ATTENTION_RES(27)
    ATTENTION_RES(28)
    ATTENTION_RES(29)
    ATTENTION_RES(30)
    ATTENTION_RES(31)
}
