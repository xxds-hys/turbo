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
    // Register<bit<8>, bit<8>>(255) offset_reg;
    // RegisterAction<bit<8>, bit<8>, bit<8>>(offset_reg) get_cur_offset = {
    //     void apply(inout bit<8> value, out bit<8> res) {
    //         if (value == 19) {
    //             value = 4;
    //             res = 19;
    //         } else {
    //             res = value;
    //             value = value + 1;
    //         }
    //     }
    // };

    // ig_intr_md.ingress_port

    // define registers  返回最大值
    Register<bit<8>, bit<8>>(255) gamma_4;
    RegisterAction<bit<8>, bit<8>, bit<8>>(gamma_4) get_gamma4_act = {
        void apply(inout bit<8> value, out bit<8> res) {
            res = max(hdr.attention.gamma, value);
        }
    };
    RegisterAction<bit<8>, bit<8>, bit<8>>(gamma_4) set_gamma4_act = {
        void apply(inout bit<8> value, out bit<8> res) {
            value = meta.gamma_new;
            res = 0;
        }
    };

    // define registers 返回差
    Register<bit<8>, bit<8>>(255) gamma_4_backup;
    RegisterAction<bit<8>, bit<8>, bit<8>>(gamma_4_backup) get_gamma4_backup_act = {
        void apply(inout bit<8> value, out bit<8> res) {
            res = min(hdr.attention.gamma, value);
        }
    };
    RegisterAction<bit<8>, bit<8>, bit<8>>(gamma_4_backup) set_gamma4_backup_act = {
        void apply(inout bit<8> value, out bit<8> res) {
            value = meta.gamma_new;
            res = 0;
        }
    };

    Register<bit<8>, bit<8>>(255) gamma_5;
    // RegisterAction<bit<8>, bit<8>, bit<16>>(gamma_5) get_gamma5_act = {
    //     void apply(inout bit<8> value, out bit<16> res) {
    //         if (value > hdr.attention.gamma) {
    //             res[15:8] = hdr.attention.gamma - value;
    //             res[7:0] = hdr.attention.gamma;
    //         } else {
    //             res[15:8] = value - hdr.attention.gamma;
    //             res[7:0] = value;
    //         }
    //     }
    // };
    RegisterAction<bit<8>, bit<8>, bit<8>>(gamma_5) set_gamma5_act = {
        void apply(inout bit<8> value, out bit<8> res) {
            value = meta.gamma_new;
            res = 0;
        }
    };

    // Register<bit<8>, bit<8>>(255) gamma_6;
    // RegisterAction<bit<8>, bit<8>, bit<16>>(gamma_6) get_gamma6_act = {
    //     void apply(inout bit<8> value, out bit<16> res) {
    //         if (value > hdr.attention.gamma) {
    //             res[15:8] = hdr.attention.gamma - value;
    //             res[7:0] = hdr.attention.gamma;
    //         } else {
    //             res[15:8] = value - hdr.attention.gamma;
    //             res[7:0] = value;
    //         }
    //     }
    // };
    // RegisterAction<bit<8>, bit<8>, bit<8>>(gamma_6) set_gamma6_act = {
    //     void apply(inout bit<8> value, out bit<8> res) {
    //         value = meta.gamma_new;
    //     }
    // };

    // Register<bit<8>, bit<8>>(255) gamma_7;
    // RegisterAction<bit<8>, bit<8>, bit<16>>(gamma_7) get_gamma7_act = {
    //     void apply(inout bit<8> value, out bit<16> res) {
    //         if (value > hdr.attention.gamma) {
    //             res[15:8] = hdr.attention.gamma - value;
    //             res[7:0] = hdr.attention.gamma;
    //         } else {
    //             res[15:8] = value - hdr.attention.gamma;
    //             res[7:0] = value;
    //         }
    //     }
    // };
    // RegisterAction<bit<8>, bit<8>, bit<8>>(gamma_7) set_gamma7_act = {
    //     void apply(inout bit<8> value, out bit<8> res) {
    //         value = meta.gamma_new;
    //     }
    // };

    // Register<bit<8>, bit<8>>(255) gamma_8;
    // RegisterAction<bit<8>, bit<8>, bit<16>>(gamma_8) get_gamma8_act = {
    //     void apply(inout bit<8> value, out bit<16> res) {
    //         if (value > hdr.attention.gamma) {
    //             res[15:8] = hdr.attention.gamma - value;
    //             res[7:0] = hdr.attention.gamma;
    //         } else {
    //             res[15:8] = value - hdr.attention.gamma;
    //             res[7:0] = value;
    //         }
    //     }
    // };
    // RegisterAction<bit<8>, bit<8>, bit<8>>(gamma_8) set_gamma8_act = {
    //     void apply(inout bit<8> value, out bit<8> res) {
    //         value = meta.gamma_new;
    //     }
    // };

    // Register<bit<8>, bit<8>>(255) gamma_9;
    // RegisterAction<bit<8>, bit<8>, bit<16>>(gamma_9) get_gamma9_act = {
    //     void apply(inout bit<8> value, out bit<16> res) {
    //         if (value > hdr.attention.gamma) {
    //             res[15:8] = hdr.attention.gamma - value;
    //             res[7:0] = hdr.attention.gamma;
    //         } else {
    //             res[15:8] = value - hdr.attention.gamma;
    //             res[7:0] = value;
    //         }
    //     }
    // };
    // RegisterAction<bit<8>, bit<8>, bit<8>>(gamma_9) set_gamma9_act = {
    //     void apply(inout bit<8> value, out bit<8> res) {
    //         value = meta.gamma_new;
    //     }
    // };

    // Register<bit<8>, bit<8>>(255) gamma_10;
    // RegisterAction<bit<8>, bit<8>, bit<16>>(gamma_10) get_gamma10_act = {
    //     void apply(inout bit<8> value, out bit<16> res) {
    //         if (value > hdr.attention.gamma) {
    //             res[15:8] = hdr.attention.gamma - value;
    //             res[7:0] = hdr.attention.gamma;
    //         } else {
    //             res[15:8] = value - hdr.attention.gamma;
    //             res[7:0] = value;
    //         }
    //     }
    // };
    // RegisterAction<bit<8>, bit<8>, bit<8>>(gamma_10) set_gamma10_act = {
    //     void apply(inout bit<8> value, out bit<8> res) {
    //         value = meta.gamma_new;
    //     }
    // };

    // Register<bit<8>, bit<8>>(255) gamma_11;
    // RegisterAction<bit<8>, bit<8>, bit<16>>(gamma_11) get_gamma11_act = {
    //     void apply(inout bit<8> value, out bit<16> res) {
    //         if (value > hdr.attention.gamma) {
    //             res[15:8] = hdr.attention.gamma - value;
    //             res[7:0] = hdr.attention.gamma;
    //         } else {
    //             res[15:8] = value - hdr.attention.gamma;
    //             res[7:0] = value;
    //         }
    //     }
    // };
    // RegisterAction<bit<8>, bit<8>, bit<8>>(gamma_11) set_gamma11_act = {
    //     void apply(inout bit<8> value, out bit<8> res) {
    //         value = meta.gamma_new;
    //     }
    // };

    // define tables

    action set_gamma_5_reg() {
        set_gamma5_act.execute(hdr.attention.seq_id);
    }

    action set_gamma_5(bit<8> res) {
        meta.gamma_new = res + meta.gamma_max;
    }
    table log_t_5 {
        key = {
            meta.gamma_ba : exact;
        }
        actions = {
            set_gamma_5;
        }
        size = 256;
    }

    // action set_gamma_6(bit<8> res) {
    //     meta.gamma_new = res + meta.gamma_max;
    //     set_gamma6_act.execute(hdr.attention.seq_id);
    // }
    // table log_t_6 {
    //     key = {
    //         meta.gamma_ba : exact;
    //     }
    //     actions = {
    //         set_gamma_6;
    //     }
    //     size = 256;
    // }

    // action set_gamma_7(bit<8> res) {
    //     meta.gamma_new = res + meta.gamma_max;
    //     set_gamma7_act.execute(hdr.attention.seq_id);
    // }
    // table log_t_7 {
    //     key = {
    //         meta.gamma_ba : exact;
    //     }
    //     actions = {
    //         set_gamma_7;
    //     }
    //     size = 256;
    // }

    // action set_gamma_8(bit<8> res) {
    //     meta.gamma_new = res + meta.gamma_max;
    //     set_gamma8_act.execute(hdr.attention.seq_id);
    // }
    // table log_t_8 {
    //     key = {
    //         meta.gamma_ba : exact;
    //     }
    //     actions = {
    //         set_gamma_8;
    //     }
    //     size = 256;
    // }

    // action set_gamma_9(bit<8> res) {
    //     meta.gamma_new = res + meta.gamma_max;
    //     set_gamma9_act.execute(hdr.attention.seq_id);
    // }
    // table log_t_9 {
    //     key = {
    //         meta.gamma_ba : exact;
    //     }
    //     actions = {
    //         set_gamma_9;
    //     }
    //     size = 256;
    // }

    // action set_gamma_10(bit<8> res) {
    //     meta.gamma_new = res + meta.gamma_max;
    //     set_gamma10_act.execute(hdr.attention.seq_id);
    // }
    // table log_t_10 {
    //     key = {
    //         meta.gamma_ba : exact;
    //     }
    //     actions = {
    //         set_gamma_10;
    //     }
    //     size = 256;
    // }

    // action set_gamma_11(bit<8> res) {
    //     meta.gamma_new = res + meta.gamma_max;
    //     set_gamma11_act.execute(hdr.attention.seq_id);
    // }
    // table log_t_11 {
    //     key = {
    //         meta.gamma_ba : exact;
    //     }
    //     actions = {
    //         set_gamma_11;
    //     }
    //     size = 256;
    // }


    apply {
        if (hdr.recirculate_gamma.isValid()) {
            
        } else if (hdr.recirculate_attention.isValid()) {

        } else {
            if (meta.offset == 4) {
                meta.gamma_max = get_gamma4_act.execute(hdr.attention.seq_id);
                meta.gamma_min = get_gamma4_backup_act.execute(hdr.attention.seq_id);
                meta.gamma_ba = meta.gamma_max - meta.gamma_min;
                // meta.gamma_ba = get_gamma4_backup_act.execute(hdr.attention.seq_id);
                log_t_5.apply();
                set_gamma_5_reg();
            } else if (meta.offset == 5) {
                // bit<16> res = get_gamma5_act.execute(hdr.attention.seq_id);
                // meta.gamma_ba = res[15:8];
                // meta.gamma_max = res[7:0];
                // log_t_6.apply();
            } else if (meta.offset == 6) {
                // bit<16> res = get_gamma6_act.execute(hdr.attention.seq_id);
                // meta.gamma_ba = res[15:8];
                // meta.gamma_max = res[7:0];
                // log_t_7.apply();
            } else if (meta.offset == 7) {
                // bit<16> res = get_gamma7_act.execute(hdr.attention.seq_id);
                // meta.gamma_ba = res[15:8];
                // meta.gamma_max = res[7:0];
                // log_t_8.apply();
            } else if (meta.offset == 8) {
                // bit<16> res = get_gamma8_act.execute(hdr.attention.seq_id);
                // meta.gamma_ba = res[15:8];
                // meta.gamma_max = res[7:0];
                // log_t_9.apply();
            } else if (meta.offset == 9) {
                // bit<16> res = get_gamma9_act.execute(hdr.attention.seq_id);
                // meta.gamma_ba = res[15:8];
                // meta.gamma_max = res[7:0];
                // log_t_10.apply();
            } else if (meta.offset == 10) {
                // bit<16> res = get_gamma10_act.execute(hdr.attention.seq_id);
                // meta.gamma_ba = res[15:8];
                // meta.gamma_max = res[7:0];
                // log_t_11.apply();
            } else if (meta.offset == 11) {
                
            } else if (meta.offset == 12) {
                
            } else if (meta.offset == 13) {
                
            } else if (meta.offset == 14) {
                
            } else if (meta.offset == 15) {
                
            } else if (meta.offset == 16) {
                
            } else if (meta.offset == 17) {
                
            } else if (meta.offset == 18) {
                
            } else if (meta.offset == 19) {
                
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
