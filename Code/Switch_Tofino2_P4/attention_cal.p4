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

    // Rolling-forward stage offset tracker (Stage 0)
    Register<bit<8>, bit<8>>(256) offset_reg;
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

    // |gamma_i - gamma_local| (stage group 0)
    Register<bit<8>, bit<8>>(256) gamma_for_ba_reg_0;
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

    // max(gamma_i, gamma_local) (stage group 0)
    Register<bit<8>, bit<8>>(256) gamma_for_max_reg_0;
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

    // |gamma_i - gamma_local| (stage group 1)
    Register<bit<8>, bit<8>>(256) gamma_for_ba_reg_1;
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

    // max(gamma_i, gamma_local) (stage group 1)
    Register<bit<8>, bit<8>>(256) gamma_for_max_reg_1;
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

    // g(delta_gamma) = log2(1 + 2^-delta_gamma) lookup table
    action set_gamma_log_res(bit<8> res, bit<4> cal_res) {
        meta.gamma_log_res = res + meta.gamma_max;
        meta.gamma_ba_cal_res = cal_res;
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

    #define ATTENTION_CAL_ACTION_PKT(i) \
        action set_attention_##i##_packet_res(bit<ATTENTION_WIDTH> res) { \
            meta.attention_##i##_pkt_res = res; \
        }

    #define ATTENTION_CAL_TABLE_PKT(i) \
        table attention_mul_t_##i## { \
            key = { \
                hdr.attention_data.data_##i##: exact; \
                meta.gamma_ba_cal_res: exact; \
                meta.local_gamma_big_flag: exact; \
            } \
            actions = { \
                set_attention_##i##_packet_res; \
            } \
            size = ATTENTION_CAL_TABLE_SIZE; \
        }

    #define ATTENTION_CAL_ACTION_LOCAL(i) \
        action set_attention_##i##_local_res(bit<ATTENTION_WIDTH> res) { \
            meta.attention_##i##_local_res = res + meta.attention_##i##_pkt_res; \
        }

    #define ATTENTION_CAL_TABLE_LOCAL(i) \
        table attention_mul_local_t_##i## { \
            key = { \
                meta.attention_set.data_##i##: exact; \
                meta.gamma_ba_cal_res: exact; \
                meta.local_gamma_big_flag: exact; \
            } \
            actions = { \
                set_attention_##i##_local_res; \
            } \
            size = ATTENTION_CAL_TABLE_SIZE; \
        }

    // Packet attention lookup actions (0..31)
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

    // Packet attention lookup tables (0..31)
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

    // Local attention lookup actions (0..31)
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

    // Local attention lookup tables (0..31)
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

    // Rolling-forward state registers (0..39, stride = 8 = 2 stages * 4 elems/stage)
    #define REG_DEF_0_7(i) \
        Register<bit<8>, bit<8>>(256) attention_##i##_reg; \
        RegisterAction<bit<8>, bit<8>, bit<8>>(attention_##i##_reg) get_attention_##i##_act = { \
            void apply(inout bit<8> value, out bit<8> res) { \
                res = value; \
            } \
        };

    REG_DEF_0_7(0)
    REG_DEF_0_7(1)
    REG_DEF_0_7(2)
    REG_DEF_0_7(3)
    REG_DEF_0_7(4)
    REG_DEF_0_7(5)
    REG_DEF_0_7(6)
    REG_DEF_0_7(7)

    #define REG_DEF(i, j) \
        Register<bit<8>, bit<8>>(256) attention_##i##_reg; \
        RegisterAction<bit<8>, bit<8>, bit<8>>(attention_##i##_reg) get_attention_##i##_act = { \
            void apply(inout bit<8> value, out bit<8> res) { \
                res = value; \
            } \
        }; \
        RegisterAction<bit<8>, bit<8>, bit<8>>(attention_##i##_reg) set_attention_##i##_act = { \
            void apply(inout bit<8> value) { \
                value = meta.attention_##j##_local_res; \
            } \
        };

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

    apply {
        if (hdr.recirculate_gamma.isValid()) {

        } else if (hdr.recirculate_attention.isValid()) {

        } else {
            meta.offset = get_cur_offset.execute(hdr.attention.seq_id);
            if (meta.offset == 4) {
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

                #define APPLY_ATTENTION(i, j) \
                    attention_mul_t_##i##.apply(); \
                    meta.attention_set.data_##i## = get_attention_##i##_act.execute(hdr.attention.seq_id); \
                    attention_mul_local_t_##i##.apply(); \
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
    inout egress_header_t                         hdr,
    inout egress_metadata_t                       meta,
    in    egress_intrinsic_metadata_t             eg_intr_md,
    in    egress_intrinsic_metadata_from_parser_t eg_prsr_md,
    inout egress_intrinsic_metadata_for_deparser_t eg_dprsr_md,
    inout egress_intrinsic_metadata_for_output_port_t eg_oport_md)
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
