#include <core.p4>
#include <tna.p4>
#include "headers.p4"
#include "parser.p4"

control Ingress (
    inout ingress_header_t hdr,
    inout ingress_metadata_t meta,
    in ingress_intrinsic_metadata_t ig_intr_md,
    in ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {

    // offset define
    Register<bit<8>, bit<8>>(255) offset_reg;
    RegisterAction<bit<8>, bit<8>, bit<8>>(offset_reg) get_cur_offset = {
        void apply(inout bit<8> value, out bit<8> res) {
            res = value;
            if (value == 19) {
                value = 4;
            } else {
                value = value + 1;
            }
        }
    };

    // ig_intr_md.ingress_port
    // 获得大小之差 0
    Register<bit<8>, bit<8>>(255) gamma_for_ba_reg_0;
    RegisterAction<bit<8>, bit<8>, bit<8>>(gamma_for_ba_reg_0) get_gamma_ba_0 = {
        void apply(inout bit<8> value, out bit<8> res) {
            if (hdr.attention.gamma > value) {
                res = hdr.attention.gamma - value;
            } else {
                res = value - hdr.attention.gamma;
            }
        }
    };
    RegisterAction<bit<8>, bit<8>, bit<8>>(gamma_for_ba_reg_0) set_gamma_ba_0 = {
        void apply(inout bit<8> value) {
            value = meta.gamma_log_res;
        }
    };

    // 获得较大的gamma 0
    Register<bit<8>, bit<8>>(255) gamma_for_max_reg_0;
    RegisterAction<bit<8>, bit<8>, bit<8>>(gamma_for_max_reg_0) get_gamma_max_0 = {
        void apply(inout bit<8> value, out bit<8> res) {
            res = max(hdr.attention.gamma, value);
        }
    };
    RegisterAction<bit<8>, bit<8>, bit<8>>(gamma_for_max_reg_0) set_gamma_max_0 = {
        void apply(inout bit<8> value) {
            value = meta.gamma_log_res;
        }
    };

    // 获得大小之差 1
    Register<bit<8>, bit<8>>(255) gamma_for_ba_reg_1;
    RegisterAction<bit<8>, bit<8>, bit<8>>(gamma_for_ba_reg_1) get_gamma_ba_1 = {
        void apply(inout bit<8> value, out bit<8> res) {
            if (hdr.attention.gamma > value) {
                res = hdr.attention.gamma - value;
            } else {
                res = value - hdr.attention.gamma;
            }
        }
    };
    RegisterAction<bit<8>, bit<8>, bit<8>>(gamma_for_ba_reg_1) set_gamma_ba_1 = {
        void apply(inout bit<8> value) {
            value = meta.gamma_log_res;
        }
    };

    // 获得较大的gamma 1
    Register<bit<8>, bit<8>>(255) gamma_for_max_reg_1;
    RegisterAction<bit<8>, bit<8>, bit<8>>(gamma_for_max_reg_1) get_gamma_max_1 = {
        void apply(inout bit<8> value, out bit<8> res) {
            res = max(hdr.attention.gamma, value);
        }
    };
    RegisterAction<bit<8>, bit<8>, bit<8>>(gamma_for_max_reg_1) set_gamma_max_1 = {
        void apply(inout bit<8> value) {
            value = meta.gamma_log_res;
        }
    };


    //定义log2(1+2^-gamma)表
    action set_gamma_log_res(bit<8> res, bit<6> cal_res) {
        meta.gamma_log_res = res + meta.gamma_max; // 计算出新的gamma_local?
        meta.gamma_ba_cal_res = cal_res; // 求解1/(1+2^gamma_ba)
    }
    table log2_gamma_t { 
        key = {
            meta.gamma_ba: exact;
        }
        actions = {
            set_gamma_log_res;
        }
        size = 256;
    }


    

    #define ATTENTION_CAL_ACTION_PKT(i)                                                 \
        action set_attention_##i##_packet_res(bit<8> res) {                             \
            meta.attention_##i##_pkt_res = res;                                         \
        }
    
    #define ATTENTION_CAL_TABLE_PKT(i)                                                  \
        table attention_mul_t_##i## {                                                   \
            key = {                                                                     \
                hdr.attention_data.data_##i##: exact;                                   \
                meta.gamma_ba_cal_res: exact;                                           \
                meta.local_gamma_big_flag: exact;                                       \
            }                                                                           \
            actions = {                                                                 \
                set_attention_##i##_packet_res;                                         \
            }                                                                           \
            size = ATTENTION_CAL_TABLE_SIZE;                                            \
        }                                                                               \
    
    #define ATTENTION_CAL_ACTION_LOCAL(i)                                               \
        action set_attention_##i##_local_res(bit<8> res) {                              \
            meta.attention_##i##_local_res = res + meta.attention_##i##_pkt_res;        \
        }

    // 将数据包携带的attention进行查表运算，放到meta.attention_i_pkt_res
    ATTENTION_CAL_ACTION_PKT(0)
    ATTENTION_CAL_ACTION_PKT(1)
    ATTENTION_CAL_ACTION_PKT(2)
    ATTENTION_CAL_ACTION_PKT(3)
    ATTENTION_CAL_ACTION_PKT(4)
    ATTENTION_CAL_ACTION_PKT(5)
    ATTENTION_CAL_ACTION_PKT(6)
    ATTENTION_CAL_ACTION_PKT(7)
    ATTENTION_CAL_ACTION_PKT(8)
    ATTENTION_CAL_ACTION_PKT(9)
    ATTENTION_CAL_ACTION_PKT(10)
    ATTENTION_CAL_ACTION_PKT(11)
    ATTENTION_CAL_ACTION_PKT(12)
    ATTENTION_CAL_ACTION_PKT(13)
    ATTENTION_CAL_ACTION_PKT(14)
    ATTENTION_CAL_ACTION_PKT(15)
    ATTENTION_CAL_ACTION_PKT(16)
    ATTENTION_CAL_ACTION_PKT(17)
    ATTENTION_CAL_ACTION_PKT(18)
    ATTENTION_CAL_ACTION_PKT(19)
    ATTENTION_CAL_ACTION_PKT(20)
    ATTENTION_CAL_ACTION_PKT(21)
    ATTENTION_CAL_ACTION_PKT(22)
    ATTENTION_CAL_ACTION_PKT(23)
    ATTENTION_CAL_ACTION_PKT(24)
    ATTENTION_CAL_ACTION_PKT(25)
    ATTENTION_CAL_ACTION_PKT(26)
    ATTENTION_CAL_ACTION_PKT(27)
    ATTENTION_CAL_ACTION_PKT(28)
    ATTENTION_CAL_ACTION_PKT(29)
    ATTENTION_CAL_ACTION_PKT(30)
    ATTENTION_CAL_ACTION_PKT(31)
    ATTENTION_CAL_ACTION_PKT(32)
    ATTENTION_CAL_ACTION_PKT(33)
    ATTENTION_CAL_ACTION_PKT(34)
    ATTENTION_CAL_ACTION_PKT(35)
    ATTENTION_CAL_ACTION_PKT(36)
    ATTENTION_CAL_ACTION_PKT(37)
    ATTENTION_CAL_ACTION_PKT(38)
    ATTENTION_CAL_ACTION_PKT(39)
    ATTENTION_CAL_ACTION_PKT(40)
    ATTENTION_CAL_ACTION_PKT(41)
    ATTENTION_CAL_ACTION_PKT(42)
    ATTENTION_CAL_ACTION_PKT(43)
    ATTENTION_CAL_ACTION_PKT(44)
    ATTENTION_CAL_ACTION_PKT(45)
    ATTENTION_CAL_ACTION_PKT(46)
    ATTENTION_CAL_ACTION_PKT(47)
    ATTENTION_CAL_ACTION_PKT(48)
    ATTENTION_CAL_ACTION_PKT(49)
    ATTENTION_CAL_ACTION_PKT(50)
    ATTENTION_CAL_ACTION_PKT(51)
    ATTENTION_CAL_ACTION_PKT(52)
    ATTENTION_CAL_ACTION_PKT(53)
    ATTENTION_CAL_ACTION_PKT(54)
    ATTENTION_CAL_ACTION_PKT(55)
    // ATTENTION_CAL_ACTION_PKT(56)
    // ATTENTION_CAL_ACTION_PKT(57)
    // ATTENTION_CAL_ACTION_PKT(58)
    // ATTENTION_CAL_ACTION_PKT(59)

    ATTENTION_CAL_TABLE_PKT(0)
    ATTENTION_CAL_TABLE_PKT(1)
    ATTENTION_CAL_TABLE_PKT(2)
    ATTENTION_CAL_TABLE_PKT(3)
    ATTENTION_CAL_TABLE_PKT(4)
    ATTENTION_CAL_TABLE_PKT(5)
    ATTENTION_CAL_TABLE_PKT(6)
    ATTENTION_CAL_TABLE_PKT(7)
    ATTENTION_CAL_TABLE_PKT(8)
    ATTENTION_CAL_TABLE_PKT(9)
    ATTENTION_CAL_TABLE_PKT(10)
    ATTENTION_CAL_TABLE_PKT(11)
    ATTENTION_CAL_TABLE_PKT(12)
    ATTENTION_CAL_TABLE_PKT(13)
    ATTENTION_CAL_TABLE_PKT(14)
    ATTENTION_CAL_TABLE_PKT(15)
    ATTENTION_CAL_TABLE_PKT(16)
    ATTENTION_CAL_TABLE_PKT(17)
    ATTENTION_CAL_TABLE_PKT(18)
    ATTENTION_CAL_TABLE_PKT(19)
    ATTENTION_CAL_TABLE_PKT(20)
    ATTENTION_CAL_TABLE_PKT(21)
    ATTENTION_CAL_TABLE_PKT(22)
    ATTENTION_CAL_TABLE_PKT(23)
    ATTENTION_CAL_TABLE_PKT(24)
    ATTENTION_CAL_TABLE_PKT(25)
    ATTENTION_CAL_TABLE_PKT(26)
    ATTENTION_CAL_TABLE_PKT(27)
    ATTENTION_CAL_TABLE_PKT(28)
    ATTENTION_CAL_TABLE_PKT(29)
    ATTENTION_CAL_TABLE_PKT(30)
    ATTENTION_CAL_TABLE_PKT(31)
    ATTENTION_CAL_TABLE_PKT(32)
    ATTENTION_CAL_TABLE_PKT(33)
    ATTENTION_CAL_TABLE_PKT(34)
    ATTENTION_CAL_TABLE_PKT(35)
    ATTENTION_CAL_TABLE_PKT(36)
    ATTENTION_CAL_TABLE_PKT(37)
    ATTENTION_CAL_TABLE_PKT(38)
    ATTENTION_CAL_TABLE_PKT(39)
    ATTENTION_CAL_TABLE_PKT(40)
    ATTENTION_CAL_TABLE_PKT(41)
    ATTENTION_CAL_TABLE_PKT(42)
    ATTENTION_CAL_TABLE_PKT(43)
    ATTENTION_CAL_TABLE_PKT(44)
    ATTENTION_CAL_TABLE_PKT(45)
    ATTENTION_CAL_TABLE_PKT(46)
    ATTENTION_CAL_TABLE_PKT(47)
    ATTENTION_CAL_TABLE_PKT(48)
    ATTENTION_CAL_TABLE_PKT(49)
    ATTENTION_CAL_TABLE_PKT(50)
    ATTENTION_CAL_TABLE_PKT(51)
    ATTENTION_CAL_TABLE_PKT(52)
    ATTENTION_CAL_TABLE_PKT(53)
    ATTENTION_CAL_TABLE_PKT(54)
    ATTENTION_CAL_TABLE_PKT(55)
    // ATTENTION_CAL_TABLE_PKT(56)
    // ATTENTION_CAL_TABLE_PKT(57)
    // ATTENTION_CAL_TABLE_PKT(58)
    // ATTENTION_CAL_TABLE_PKT(59)

    // 更新本地的attention到metadata
    ATTENTION_CAL_ACTION_LOCAL(0)
    ATTENTION_CAL_ACTION_LOCAL(1)
    ATTENTION_CAL_ACTION_LOCAL(2)
    ATTENTION_CAL_ACTION_LOCAL(3)
    ATTENTION_CAL_ACTION_LOCAL(4)
    ATTENTION_CAL_ACTION_LOCAL(5)
    ATTENTION_CAL_ACTION_LOCAL(6)
    ATTENTION_CAL_ACTION_LOCAL(7)
    ATTENTION_CAL_ACTION_LOCAL(8)
    ATTENTION_CAL_ACTION_LOCAL(9)
    ATTENTION_CAL_ACTION_LOCAL(10)
    ATTENTION_CAL_ACTION_LOCAL(11)
    ATTENTION_CAL_ACTION_LOCAL(12)
    ATTENTION_CAL_ACTION_LOCAL(13)
    ATTENTION_CAL_ACTION_LOCAL(14)
    ATTENTION_CAL_ACTION_LOCAL(15)
    ATTENTION_CAL_ACTION_LOCAL(16)
    ATTENTION_CAL_ACTION_LOCAL(17)
    ATTENTION_CAL_ACTION_LOCAL(18)
    ATTENTION_CAL_ACTION_LOCAL(19)
    ATTENTION_CAL_ACTION_LOCAL(20)
    ATTENTION_CAL_ACTION_LOCAL(21)
    ATTENTION_CAL_ACTION_LOCAL(22)
    ATTENTION_CAL_ACTION_LOCAL(23)
    ATTENTION_CAL_ACTION_LOCAL(24)
    ATTENTION_CAL_ACTION_LOCAL(25)
    ATTENTION_CAL_ACTION_LOCAL(26)
    ATTENTION_CAL_ACTION_LOCAL(27)
    ATTENTION_CAL_ACTION_LOCAL(28)
    ATTENTION_CAL_ACTION_LOCAL(29)
    ATTENTION_CAL_ACTION_LOCAL(30)
    ATTENTION_CAL_ACTION_LOCAL(31)
    ATTENTION_CAL_ACTION_LOCAL(32)
    ATTENTION_CAL_ACTION_LOCAL(33)
    ATTENTION_CAL_ACTION_LOCAL(34)
    ATTENTION_CAL_ACTION_LOCAL(35)
    ATTENTION_CAL_ACTION_LOCAL(36)
    ATTENTION_CAL_ACTION_LOCAL(37)
    ATTENTION_CAL_ACTION_LOCAL(38)
    ATTENTION_CAL_ACTION_LOCAL(39)
    ATTENTION_CAL_ACTION_LOCAL(40)
    ATTENTION_CAL_ACTION_LOCAL(41)
    ATTENTION_CAL_ACTION_LOCAL(42)
    ATTENTION_CAL_ACTION_LOCAL(43)
    ATTENTION_CAL_ACTION_LOCAL(44)
    ATTENTION_CAL_ACTION_LOCAL(45)
    ATTENTION_CAL_ACTION_LOCAL(46)
    ATTENTION_CAL_ACTION_LOCAL(47)
    ATTENTION_CAL_ACTION_LOCAL(48)
    ATTENTION_CAL_ACTION_LOCAL(49)
    ATTENTION_CAL_ACTION_LOCAL(50)
    ATTENTION_CAL_ACTION_LOCAL(51)
    ATTENTION_CAL_ACTION_LOCAL(52)
    ATTENTION_CAL_ACTION_LOCAL(53)
    ATTENTION_CAL_ACTION_LOCAL(54)
    ATTENTION_CAL_ACTION_LOCAL(55)
    // ATTENTION_CAL_ACTION_LOCAL(56)
    // ATTENTION_CAL_ACTION_LOCAL(57)
    // ATTENTION_CAL_ACTION_LOCAL(58)
    // ATTENTION_CAL_ACTION_LOCAL(59)

    #define ATTENTION_CAL_TABLE_LOCAL(i)                               \
    table attention_mul_local_t_##i## {                                 \
        key = {                                                         \
            meta.attention_set.data_##i##: exact;                       \
            meta.gamma_ba_cal_res: exact;                               \
            meta.local_gamma_big_flag: exact;                           \
        }                                                               \
        actions = {                                                     \
            set_attention_##i##_local_res;                              \
        }                                                               \
        size = ATTENTION_CAL_TABLE_SIZE;                                \
    }

    ATTENTION_CAL_TABLE_LOCAL(0)
    ATTENTION_CAL_TABLE_LOCAL(1)
    ATTENTION_CAL_TABLE_LOCAL(2)
    ATTENTION_CAL_TABLE_LOCAL(3)
    ATTENTION_CAL_TABLE_LOCAL(4)
    ATTENTION_CAL_TABLE_LOCAL(5)
    ATTENTION_CAL_TABLE_LOCAL(6)
    ATTENTION_CAL_TABLE_LOCAL(7)
    ATTENTION_CAL_TABLE_LOCAL(8)
    ATTENTION_CAL_TABLE_LOCAL(9)
    ATTENTION_CAL_TABLE_LOCAL(10)
    ATTENTION_CAL_TABLE_LOCAL(11)
    ATTENTION_CAL_TABLE_LOCAL(12)
    ATTENTION_CAL_TABLE_LOCAL(13)
    ATTENTION_CAL_TABLE_LOCAL(14)
    ATTENTION_CAL_TABLE_LOCAL(15)
    ATTENTION_CAL_TABLE_LOCAL(16)
    ATTENTION_CAL_TABLE_LOCAL(17)
    ATTENTION_CAL_TABLE_LOCAL(18)
    ATTENTION_CAL_TABLE_LOCAL(19)
    ATTENTION_CAL_TABLE_LOCAL(20)
    ATTENTION_CAL_TABLE_LOCAL(21)
    ATTENTION_CAL_TABLE_LOCAL(22)
    ATTENTION_CAL_TABLE_LOCAL(23)
    ATTENTION_CAL_TABLE_LOCAL(24)
    ATTENTION_CAL_TABLE_LOCAL(25)
    ATTENTION_CAL_TABLE_LOCAL(26)
    ATTENTION_CAL_TABLE_LOCAL(27)
    ATTENTION_CAL_TABLE_LOCAL(28)
    ATTENTION_CAL_TABLE_LOCAL(29)
    ATTENTION_CAL_TABLE_LOCAL(30)
    ATTENTION_CAL_TABLE_LOCAL(31)
    ATTENTION_CAL_TABLE_LOCAL(32)
    ATTENTION_CAL_TABLE_LOCAL(33)
    ATTENTION_CAL_TABLE_LOCAL(34)
    ATTENTION_CAL_TABLE_LOCAL(35)
    ATTENTION_CAL_TABLE_LOCAL(36)
    ATTENTION_CAL_TABLE_LOCAL(37)
    ATTENTION_CAL_TABLE_LOCAL(38)
    ATTENTION_CAL_TABLE_LOCAL(39)
    ATTENTION_CAL_TABLE_LOCAL(40)
    ATTENTION_CAL_TABLE_LOCAL(41)
    ATTENTION_CAL_TABLE_LOCAL(42)
    ATTENTION_CAL_TABLE_LOCAL(43)
    ATTENTION_CAL_TABLE_LOCAL(44)
    ATTENTION_CAL_TABLE_LOCAL(45)
    ATTENTION_CAL_TABLE_LOCAL(46)
    ATTENTION_CAL_TABLE_LOCAL(47)
    ATTENTION_CAL_TABLE_LOCAL(48)
    ATTENTION_CAL_TABLE_LOCAL(49)
    ATTENTION_CAL_TABLE_LOCAL(50)
    ATTENTION_CAL_TABLE_LOCAL(51)
    ATTENTION_CAL_TABLE_LOCAL(52)
    ATTENTION_CAL_TABLE_LOCAL(53)
    ATTENTION_CAL_TABLE_LOCAL(54)
    ATTENTION_CAL_TABLE_LOCAL(55)

    // @pragma stage(3)
    // table attention_mul_local_t_0 {
    //     key = {
    //         meta.attention_set.data_0: exact;
    //         meta.gamma_ba_cal_res: exact;
    //         meta.local_gamma_big_flag: exact;
    //     }
    //     actions = {
    //         set_attention_0_local_res;
    //     }
    //     size = ATTENTION_CAL_TABLE_SIZE;
    // }

    // @pragma stage(3)
    // table attention_mul_local_t_1 {
    //     key = {
    //         meta.attention_set.data_1: exact;
    //         meta.gamma_ba_cal_res: exact;
    //         meta.local_gamma_big_flag: exact;
    //     }
    //     actions = {
    //         set_attention_1_local_res;
    //     }
    //     size = ATTENTION_CAL_TABLE_SIZE;
    // }

    // @pragma stage(3)
    // table attention_mul_local_t_2 {
    //     key = {
    //         meta.attention_set.data_2: exact;
    //         meta.gamma_ba_cal_res: exact;
    //         meta.local_gamma_big_flag: exact;
    //     }
    //     actions = {
    //         set_attention_2_local_res;
    //     }
    //     size = ATTENTION_CAL_TABLE_SIZE;
    // }

    // @pragma stage(3)
    // table attention_mul_local_t_3 {
    //     key = {
    //         meta.attention_set.data_3[5:0]: exact;
    //         meta.gamma_ba_cal_res: exact;
    //         meta.local_gamma_big_flag: exact;
    //     }
    //     actions = {
    //         set_attention_3_local_res;
    //     }
    //     size = ATTENTION_CAL_TABLE_SIZE;
    // }

    // @pragma stage(4)
    // table attention_mul_local_t_4 {
    //     key = {
    //         meta.attention_set.data_4[5:0]: exact;
    //         meta.gamma_ba_cal_res: exact;
    //         meta.local_gamma_big_flag: exact;
    //     }
    //     actions = {
    //         set_attention_4_local_res;
    //     }
    //     size = ATTENTION_CAL_TABLE_SIZE;
    // }


    // @pragma stage(4)
    // table attention_mul_local_t_5 {           
    //     key = {                                   
    //         meta.attention_set.data_5[5:0]: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_5_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(4)
    // table attention_mul_local_t_6 {           
    //     key = {                                   
    //         meta.attention_set.data_6[5:0]: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_6_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(4)
    // table attention_mul_local_t_7 {           
    //     key = {                                   
    //         meta.attention_set.data_7[5:0]: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_7_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(5)
    // table attention_mul_local_t_8 {           
    //     key = {                                   
    //         meta.attention_set.data_8[5:0]: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_8_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(5)
    // table attention_mul_local_t_9 {           
    //     key = {                                   
    //         meta.attention_set.data_9[5:0]: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_9_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(5)
    // table attention_mul_local_t_10 {
    //     key = {
    //         meta.attention_set.data_10[5:0]: exact;
    //         meta.gamma_ba_cal_res: exact;
    //         meta.local_gamma_big_flag: exact;
    //     }
    //     actions = {
    //         set_attention_10_local_res;
    //     }
    //     size = ATTENTION_CAL_TABLE_SIZE;
    // }

    // @pragma stage(5)
    // table attention_mul_local_t_11 {
    //     key = {
    //         meta.attention_set.data_11[5:0]: exact;
    //         meta.gamma_ba_cal_res: exact;
    //         meta.local_gamma_big_flag: exact;
    //     }
    //     actions = {
    //         set_attention_11_local_res;
    //     }
    //     size = ATTENTION_CAL_TABLE_SIZE;
    // }

    // @pragma stage(6)
    // table attention_mul_local_t_12 {
    //     key = {
    //         meta.attention_set.data_12[5:0]: exact;
    //         meta.gamma_ba_cal_res: exact;
    //         meta.local_gamma_big_flag: exact;
    //     }
    //     actions = {
    //         set_attention_12_local_res;
    //     }
    //     size = ATTENTION_CAL_TABLE_SIZE;
    // }

    // @pragma stage(6)
    // table attention_mul_local_t_13 {
    //     key = {
    //         meta.attention_set.data_13[5:0]: exact;
    //         meta.gamma_ba_cal_res: exact;
    //         meta.local_gamma_big_flag: exact;
    //     }
    //     actions = {
    //         set_attention_13_local_res;
    //     }
    //     size = ATTENTION_CAL_TABLE_SIZE;
    // }

    // @pragma stage(6)
    // table attention_mul_local_t_14 {
    //     key = {
    //         meta.attention_set.data_14[5:0]: exact;
    //         meta.gamma_ba_cal_res: exact;
    //         meta.local_gamma_big_flag: exact;
    //     }
    //     actions = {
    //         set_attention_14_local_res;
    //     }
    //     size = ATTENTION_CAL_TABLE_SIZE;
    // }


    // @pragma stage(6)
    // table attention_mul_local_t_15 {           
    //     key = {                                   
    //         meta.attention_set.data_15[5:0]: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_15_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(7)
    // table attention_mul_local_t_16 {           
    //     key = {                                   
    //         meta.attention_set.data_16[5:0]: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_16_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(7)
    // table attention_mul_local_t_17 {           
    //     key = {                                   
    //         meta.attention_set.data_17[5:0]: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_17_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(7)
    // table attention_mul_local_t_18 {           
    //     key = {                                   
    //         meta.attention_set.data_18[5:0]: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_18_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(7)
    // table attention_mul_local_t_19 {           
    //     key = {                                   
    //         meta.attention_set.data_19[5:0]: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_19_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(8)
    // table attention_mul_local_t_20 {
    //     key = {
    //         meta.attention_set.data_20[5:0]: exact;
    //         meta.gamma_ba_cal_res: exact;
    //         meta.local_gamma_big_flag: exact;
    //     }
    //     actions = {
    //         set_attention_20_local_res;
    //     }
    //     size = ATTENTION_CAL_TABLE_SIZE;
    // }

    // @pragma stage(8)
    // table attention_mul_local_t_21 {
    //     key = {
    //         meta.attention_set.data_21[5:0]: exact;
    //         meta.gamma_ba_cal_res: exact;
    //         meta.local_gamma_big_flag: exact;
    //     }
    //     actions = {
    //         set_attention_21_local_res;
    //     }
    //     size = ATTENTION_CAL_TABLE_SIZE;
    // }

    // @pragma stage(8)
    // table attention_mul_local_t_22 {
    //     key = {
    //         meta.attention_set.data_22[5:0]: exact;
    //         meta.gamma_ba_cal_res: exact;
    //         meta.local_gamma_big_flag: exact;
    //     }
    //     actions = {
    //         set_attention_22_local_res;
    //     }
    //     size = ATTENTION_CAL_TABLE_SIZE;
    // }

    // @pragma stage(8)
    // table attention_mul_local_t_23 {
    //     key = {
    //         meta.attention_set.data_23[5:0]: exact;
    //         meta.gamma_ba_cal_res: exact;
    //         meta.local_gamma_big_flag: exact;
    //     }
    //     actions = {
    //         set_attention_23_local_res;
    //     }
    //     size = ATTENTION_CAL_TABLE_SIZE;
    // }

    // @pragma stage(9)
    // table attention_mul_local_t_24 {
    //     key = {
    //         meta.attention_set.data_24[5:0]: exact;
    //         meta.gamma_ba_cal_res: exact;
    //         meta.local_gamma_big_flag: exact;
    //     }
    //     actions = {
    //         set_attention_24_local_res;
    //     }
    //     size = ATTENTION_CAL_TABLE_SIZE;
    // }


    // @pragma stage(9)
    // table attention_mul_local_t_25 {           
    //     key = {                                   
    //         meta.attention_set.data_25[5:0]: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_25_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(9)
    // table attention_mul_local_t_26 {           
    //     key = {                                   
    //         meta.attention_set.data_26[5:0]: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_26_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(9)
    // table attention_mul_local_t_27 {           
    //     key = {                                   
    //         meta.attention_set.data_27[5:0]: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_27_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(10)
    // table attention_mul_local_t_28 {           
    //     key = {                                   
    //         meta.attention_set.data_28[5:0]: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_28_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(10)
    // table attention_mul_local_t_29 {           
    //     key = {                                   
    //         meta.attention_set.data_29[5:0]: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_29_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(10)
    // table attention_mul_local_t_30 {           
    //     key = {                                   
    //         meta.attention_set.data_30[5:0]: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_30_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(10)
    // table attention_mul_local_t_31 {           
    //     key = {                                   
    //         meta.attention_set.data_31[5:0]: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_31_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(11)
    // table attention_mul_local_t_32 {
    //     key = {
    //         meta.attention_set.data_32: exact;
    //         meta.gamma_ba_cal_res: exact;
    //         meta.local_gamma_big_flag: exact;
    //     }
    //     actions = {
    //         set_attention_32_local_res;
    //     }
    //     size = ATTENTION_CAL_TABLE_SIZE;
    // }

    // @pragma stage(11)
    // table attention_mul_local_t_33 {
    //     key = {
    //         meta.attention_set.data_33: exact;
    //         meta.gamma_ba_cal_res: exact;
    //         meta.local_gamma_big_flag: exact;
    //     }
    //     actions = {
    //         set_attention_33_local_res;
    //     }
    //     size = ATTENTION_CAL_TABLE_SIZE;
    // }

    // @pragma stage(11)
    // table attention_mul_local_t_34 {
    //     key = {
    //         meta.attention_set.data_34: exact;
    //         meta.gamma_ba_cal_res: exact;
    //         meta.local_gamma_big_flag: exact;
    //     }
    //     actions = {
    //         set_attention_34_local_res;
    //     }
    //     size = ATTENTION_CAL_TABLE_SIZE;
    // }


    // @pragma stage(11)
    // table attention_mul_local_t_35 {           
    //     key = {                                   
    //         meta.attention_set.data_35: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_35_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(12)
    // table attention_mul_local_t_36 {           
    //     key = {                                   
    //         meta.attention_set.data_36: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_36_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(12)
    // table attention_mul_local_t_37 {           
    //     key = {                                   
    //         meta.attention_set.data_37: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_37_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(12)
    // table attention_mul_local_t_38 {           
    //     key = {                                   
    //         meta.attention_set.data_38: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_38_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(12)
    // table attention_mul_local_t_39 {           
    //     key = {                                   
    //         meta.attention_set.data_39: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_39_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(13)
    // table attention_mul_local_t_40 {           
    //     key = {                                   
    //         meta.attention_set.data_40: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_40_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(13)
    // table attention_mul_local_t_41 {           
    //     key = {                                   
    //         meta.attention_set.data_41: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_41_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(13)
    // table attention_mul_local_t_42 {
    //     key = {
    //         meta.attention_set.data_42: exact;
    //         meta.gamma_ba_cal_res: exact;
    //         meta.local_gamma_big_flag: exact;
    //     }
    //     actions = {
    //         set_attention_42_local_res;
    //     }
    //     size = ATTENTION_CAL_TABLE_SIZE;
    // }

    // @pragma stage(13)
    // table attention_mul_local_t_43 {
    //     key = {
    //         meta.attention_set.data_43: exact;
    //         meta.gamma_ba_cal_res: exact;
    //         meta.local_gamma_big_flag: exact;
    //     }
    //     actions = {
    //         set_attention_43_local_res;
    //     }
    //     size = ATTENTION_CAL_TABLE_SIZE;
    // }

    // @pragma stage(14)
    // table attention_mul_local_t_44 {
    //     key = {
    //         meta.attention_set.data_44: exact;
    //         meta.gamma_ba_cal_res: exact;
    //         meta.local_gamma_big_flag: exact;
    //     }
    //     actions = {
    //         set_attention_44_local_res;
    //     }
    //     size = ATTENTION_CAL_TABLE_SIZE;
    // }


    // @pragma stage(14)
    // table attention_mul_local_t_45 {           
    //     key = {                                   
    //         meta.attention_set.data_45: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_45_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(14)
    // table attention_mul_local_t_46 {           
    //     key = {                                   
    //         meta.attention_set.data_46: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_46_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(14)
    // table attention_mul_local_t_47 {           
    //     key = {                                   
    //         meta.attention_set.data_47: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_47_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(15)
    // table attention_mul_local_t_48 {           
    //     key = {                                   
    //         meta.attention_set.data_48: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_48_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(15)
    // table attention_mul_local_t_49 {           
    //     key = {                                   
    //         meta.attention_set.data_49: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_49_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(15)
    // table attention_mul_local_t_50 {           
    //     key = {                                   
    //         meta.attention_set.data_50: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_50_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(15)
    // table attention_mul_local_t_51 {           
    //     key = {                                   
    //         meta.attention_set.data_51: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_51_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }

    // @pragma stage(16)
    // table attention_mul_local_t_52 {
    //     key = {
    //         meta.attention_set.data_52: exact;
    //         meta.gamma_ba_cal_res: exact;
    //         meta.local_gamma_big_flag: exact;
    //     }
    //     actions = {
    //         set_attention_52_local_res;
    //     }
    //     size = ATTENTION_CAL_TABLE_SIZE;
    // }

    // @pragma stage(16)
    // table attention_mul_local_t_53 {
    //     key = {
    //         meta.attention_set.data_53: exact;
    //         meta.gamma_ba_cal_res: exact;
    //         meta.local_gamma_big_flag: exact;
    //     }
    //     actions = {
    //         set_attention_53_local_res;
    //     }
    //     size = ATTENTION_CAL_TABLE_SIZE;
    // }

    // @pragma stage(15)
    // table attention_mul_local_t_54 {
    //     key = {
    //         meta.attention_set.data_54: exact;
    //         meta.gamma_ba_cal_res: exact;
    //         meta.local_gamma_big_flag: exact;
    //     }
    //     actions = {
    //         set_attention_54_local_res;
    //     }
    //     size = ATTENTION_CAL_TABLE_SIZE;
    // }


    // @pragma stage(15)
    // table attention_mul_local_t_55 {           
    //     key = {                                   
    //         meta.attention_set.data_55: exact; 
    //         meta.gamma_ba_cal_res: exact;         
    //         meta.local_gamma_big_flag: exact;     
    //     }                                         
    //     actions = {                               
    //         set_attention_55_local_res;        
    //     }                                         
    //     size = ATTENTION_CAL_TABLE_SIZE;          
    // }


    #define REG_DEF_0_4(i)                                                                      \
    Register<bit<8>, bit<8>>(256) attention_##i##_reg;                                          \
    RegisterAction<bit<8>, bit<8>, bit<8>>(attention_##i##_reg) get_attention_##i##_act = {     \
        void apply(inout bit<8> value, out bit<8> res) {                                        \
            res = value;                                                                        \
        }                                                                                       \
    };
    
    REG_DEF_0_4(0)
    REG_DEF_0_4(1)
    REG_DEF_0_4(2)
    REG_DEF_0_4(3)
    REG_DEF_0_4(4)
    REG_DEF_0_4(5)
    REG_DEF_0_4(6)
    REG_DEF_0_4(7)


    #define REG_DEF(i, j)                                                                           \
        Register<bit<8>, bit<8>>(256) attention_##i##_reg;                                          \
        RegisterAction<bit<8>, bit<8>, bit<8>>(attention_##i##_reg) get_attention_##i##_act = {     \
            void apply(inout bit<8> value, out bit<8> res) {                                        \
                res = value;                                                                        \
            }                                                                                       \
        };                                                                                          \
        RegisterAction<bit<8>, bit<8>, bit<8>>(attention_##i##_reg) set_attention_##i##_act = {     \
            void apply(inout bit<8> value) {                                                        \
                value = meta.attention_##j##_local_res;                                             \
            }                                                                                       \
        };
    
    /************************* Begin DirectRegister *********************************/
    DirectRegister<bit<8>>() attention_register_example;
    DirectRegisterAction<bit<8>, bit<8>>(attention_register_example) get_attention_reg_exmaple = {
        void apply(inout bit<8> value, out bit<8> res) {
            res = value;
        }
    };
    DirectRegisterAction<bit<8>, bit<8>>(attention_register_example) set_attention_reg_exmaple = {
        void apply(inout bit<8> value) {
            value = 2;
        }
    };

    action get_attention_act_example() {
        get_attention_reg_exmaple.execute();
    }
    /*************************End DirectRegister *********************************/

    // action set_attention_act_example() {
    //     set_attention_reg_exmaple.execute();
    // }

    table attention_example_t {
        key = {
            hdr.attention.seq_id: exact;
        }
        actions = {
            get_attention_act_example;
            // set_attention_act_example;
        }
        size = 256;
        registers = attention_register_example;
    }
    
    REG_DEF(8, 0)
    REG_DEF(9, 1)
    REG_DEF(10, 2)
    REG_DEF(11, 3)
    REG_DEF(12, 4)
    REG_DEF(13, 5)
    REG_DEF(14, 6)
    REG_DEF(15, 7)

    REG_DEF(16, 8)
    REG_DEF(17, 9)
    REG_DEF(18, 10)
    REG_DEF(19, 11)
    REG_DEF(20, 12)
    REG_DEF(21, 13)
    REG_DEF(22, 14)
    REG_DEF(23, 15)

    REG_DEF(24, 16)
    REG_DEF(25, 17)
    REG_DEF(26, 18)
    REG_DEF(27, 19)
    REG_DEF(28, 20)
    REG_DEF(29, 21)
    REG_DEF(30, 22)
    REG_DEF(31, 23)

    REG_DEF(32, 24)
    REG_DEF(33, 25)
    REG_DEF(34, 26)
    REG_DEF(35, 27)
    REG_DEF(36, 28)
    REG_DEF(37, 29)
    REG_DEF(38, 30)
    REG_DEF(39, 31)

    REG_DEF(40, 32)
    REG_DEF(41, 33)
    REG_DEF(42, 34)
    REG_DEF(43, 35)
    REG_DEF(44, 36)
    REG_DEF(45, 37)
    REG_DEF(46, 38)
    REG_DEF(47, 39)

    REG_DEF(48, 40)
    REG_DEF(49, 41)
    REG_DEF(50, 42)
    REG_DEF(51, 43)
    REG_DEF(52, 44)
    REG_DEF(53, 45)
    REG_DEF(54, 46)
    REG_DEF(55, 47)

    REG_DEF(56, 48)
    REG_DEF(57, 49)
    REG_DEF(58, 50)
    REG_DEF(59, 51)
    REG_DEF(60, 52)
    REG_DEF(61, 53)
    REG_DEF(62, 54)
    REG_DEF(63, 55)

    apply {
        if (hdr.recirculate_gamma.isValid()) {
            
        } else if (hdr.recirculate_attention.isValid()) {

        } else {
            meta.offset = get_cur_offset.execute(hdr.attention.seq_id);
            if (meta.offset == 2) {
                attention_example_t.apply();
            } else if (meta.offset == 4) {
                meta.gamma_ba = get_gamma_ba_0.execute(hdr.attention.seq_id);
                meta.gamma_max = get_gamma_max_0.execute(hdr.attention.seq_id);
                log2_gamma_t.apply();
                set_gamma_ba_1.execute(hdr.attention.seq_id);
                set_gamma_max_1.execute(hdr.attention.seq_id);
                if (hdr.attention.gamma == meta.gamma_max) {
                    meta.local_gamma_big_flag = 0;
                } else {
                    meta.local_gamma_big_flag = 1;
                }

                /****** attention 0 的计算 ********/
                // // 计算数据包携带的attention
                // attention_mul_t_0.apply();
                // // 得到本地的attention 0
                // meta.attention_set.data_0 = get_attention_0_act.execute(hdr.attention.seq_id);
                // // 查表计算本地的attention 0
                // attention_mul_local_t_0.apply();
                // // 写回attention
                // set_attention_5_act.execute(hdr.attention.seq_id);

                #define APPLY_ATTENTION(i, j)                                                           \
                attention_mul_t_##i##.apply();                                                          \
                meta.attention_set.data_##i## = get_attention_##i##_act.execute(hdr.attention.seq_id);  \
                attention_mul_local_t_##i##.apply();                                                    \
                set_attention_##j##_act.execute(hdr.attention.seq_id);

                APPLY_ATTENTION(0, 8)
                APPLY_ATTENTION(1, 9)
                APPLY_ATTENTION(2, 10)
                APPLY_ATTENTION(3, 11)
                APPLY_ATTENTION(4, 12)
                APPLY_ATTENTION(5, 13)
                APPLY_ATTENTION(6, 14)
                APPLY_ATTENTION(7, 15)
                APPLY_ATTENTION(8, 16)
                APPLY_ATTENTION(9, 17)
                APPLY_ATTENTION(10, 18)
                APPLY_ATTENTION(11, 19)
                APPLY_ATTENTION(12, 20)
                APPLY_ATTENTION(13, 21)
                APPLY_ATTENTION(14, 22)
                APPLY_ATTENTION(15, 23)
                APPLY_ATTENTION(16, 24)
                APPLY_ATTENTION(17, 25)
                APPLY_ATTENTION(18, 26)
                APPLY_ATTENTION(19, 27)
                APPLY_ATTENTION(20, 28)
                APPLY_ATTENTION(21, 29)
                APPLY_ATTENTION(22, 30)
                APPLY_ATTENTION(23, 31)
                APPLY_ATTENTION(24, 32)
                APPLY_ATTENTION(25, 33)
                APPLY_ATTENTION(26, 34)
                APPLY_ATTENTION(27, 35)
                APPLY_ATTENTION(28, 36)
                APPLY_ATTENTION(29, 37)
                APPLY_ATTENTION(30, 38)
                APPLY_ATTENTION(31, 39)
                APPLY_ATTENTION(32, 40)
                APPLY_ATTENTION(33, 41)
                APPLY_ATTENTION(34, 42)
                APPLY_ATTENTION(35, 43)
                APPLY_ATTENTION(36, 44)
                APPLY_ATTENTION(37, 45)
                APPLY_ATTENTION(38, 46)
                APPLY_ATTENTION(39, 47)
                APPLY_ATTENTION(40, 48)
                APPLY_ATTENTION(41, 49)
                APPLY_ATTENTION(42, 50)
                APPLY_ATTENTION(43, 51)
                APPLY_ATTENTION(44, 52)
                APPLY_ATTENTION(45, 53)
                APPLY_ATTENTION(46, 54)
                APPLY_ATTENTION(47, 55)
                APPLY_ATTENTION(48, 56)
                APPLY_ATTENTION(49, 57)
                APPLY_ATTENTION(50, 58)
                APPLY_ATTENTION(51, 59)
                APPLY_ATTENTION(52, 60)
                APPLY_ATTENTION(53, 61)
                APPLY_ATTENTION(54, 62)
                APPLY_ATTENTION(55, 63)

            } else if (meta.offset == 6) {
                
            } else if (meta.offset == 8) {
                
            } else if (meta.offset == 10) {
                
            } else if (meta.offset == 12) {
                
            } else if (meta.offset == 14) {
                
            } else if (meta.offset == 16) {
                
            } else if (meta.offset == 18) {
                
            } else {
                // nothing, pass
            }
        }
    }

}

control Egress(
    inout egress_header_t                          hdr,
    inout egress_metadata_t                         meta, 
    in    egress_intrinsic_metadata_t                  eg_intr_md,
    in    egress_intrinsic_metadata_from_parser_t      eg_prsr_md,
    inout egress_intrinsic_metadata_for_deparser_t     eg_dprsr_md,
    inout egress_intrinsic_metadata_for_output_port_t  eg_oport_md)
{
    apply {

    }
}

Pipeline(
    IngressParser(),
    Ingress(),
    IngressDeparser(),
    EgressParser(),
    Egress(),
    EgressDeparser()
) pipe;

Switch(pipe) main;