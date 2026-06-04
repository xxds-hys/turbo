
parser TofinoIngressParser(
    packet_in pkt,
    out ingress_intrinsic_metadata_t ig_intr_md
) 
{
    state start {
        pkt.extract(ig_intr_md);
        transition select(ig_intr_md.resubmit_flag) {
            1: parse_resubmit; // not in use
            0: parse_port_metadata;
        }
    }

    state parse_resubmit {
        transition reject;
    }

    state parse_port_metadata {
        pkt.advance(PORT_METADATA_SIZE);
        transition accept;
    }
}

parser TofinoEgressParser(
        packet_in pkt,
        out egress_intrinsic_metadata_t eg_intr_md
) {
    state start {
        pkt.extract(eg_intr_md);
        transition accept;
    }
}

parser IngressParser (
    packet_in pkt,
    out ingress_header_t hdr,
    out ingress_metadata_t ig_md,
    out ingress_intrinsic_metadata_t ig_intr_md
) {
    TofinoIngressParser() tofino_parser;

    state start {
        tofino_parser.apply(pkt, ig_intr_md);
        transition check_recirculate;
    }

    state check_recirculate {
        bit<8> recirculate_id = pkt.lookahead<bit<8>>();
        transition select(recirculate_id) {
            220: parse_recirculate;
            _: parse_ethernet;
        }
    }

    state parse_recirculate {
        pkt.extract(hdr.recirculate);
        transition select(hdr.recirculate.recirculate_tag) {
            0: parse_recirculate_gamma;
            1: parse_recirculate_attention;
        }
    }

    state parse_recirculate_gamma {
        pkt.extract(hdr.recirculate_gamma);
        transition parse_ethernet;
    }

    state parse_recirculate_attention {
        pkt.extract(hdr.recirculate_attention);
        transition parse_ethernet;
    }

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            0x0800: parse_ipv4;
            default: accept;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            17: parse_udp;
            default: accept;
        }
    }

    state parse_udp {
        pkt.extract(hdr.udp);
        transition select(hdr.udp.dst_port) {
            0x8888: parse_attention;
            default: accept;
        }
    }

    state parse_attention {
        pkt.extract(hdr.attention);
        pkt.extract(hdr.attention_data);
        transition accept;
    }
}

control IngressDeparser(
    packet_out pkt,
    inout ingress_header_t hdr,
    in ingress_metadata_t ig_md,
    in ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md
) {
    apply {
        pkt.emit(hdr.recirculate);
        pkt.emit(hdr.recirculate_gamma);
        pkt.emit(hdr.recirculate_attention);
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.ipv4);
        pkt.emit(hdr.udp);
        pkt.emit(hdr.attention);
        pkt.emit(hdr.attention_data);
    }
}

parser EgressParser(
    packet_in pkt,
    out egress_header_t hdr,
    out egress_metadata_t eg_md,
    out egress_intrinsic_metadata_t eg_intr_md
) {
    TofinoEgressParser() tofino_parser;
    state start {
        tofino_parser.apply(pkt, eg_intr_md);
        transition accept;
    }
}

control EgressDeparser(
    packet_out pkt,
    inout egress_header_t hdr,
    in egress_metadata_t eg_md,
    in egress_intrinsic_metadata_for_deparser_t eg_dprsr_md
) {
    apply{

    }
}