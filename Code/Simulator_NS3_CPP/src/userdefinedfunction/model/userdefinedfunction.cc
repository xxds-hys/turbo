/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

#include "userdefinedfunction.h"


namespace ns3{

NS_LOG_COMPONENT_DEFINE("userdefinedfunction");

void add_channel_between_two_nodes(Ptr<Node> firstNode, Ptr<Node> secondNode, PointToPointHelper &p2p) {
  NodeContainer nodes;
  nodes.Add(firstNode);
  nodes.Add(secondNode);
  p2p.Install(nodes);
}

uint32_t add_channels(NodeContainer &allNodes, std::map<uint32_t, CHL_entry_t> &CHL){
  uint32_t channelCnt = CHL.size();
  uint32_t installChannelCnt = 0;
  for (uint32_t i = 0; i < channelCnt; i++) {
    CHL_entry_t channelEntry = CHL[i];
    uint32_t srcNodeIdx = channelEntry.srcNodeIdx;
    Ptr<Node> srcNode = allNodes.Get(srcNodeIdx);
    uint32_t dstNodeIdx = channelEntry.dstNodeIdx;
    Ptr<Node> dstNode = allNodes.Get(dstNodeIdx);
    PointToPointHelper p2p = set_P2P_attribute(channelEntry.widthInGbps,
                                               channelEntry.delayInUs,
                                               channelEntry.queueType,
                                               channelEntry.queueSize);
    add_channel_between_two_nodes(srcNode, dstNode, p2p);
    installChannelCnt = installChannelCnt + 1;
  }
  return installChannelCnt;
}

void assign_address_to_single_device(Ipv4Address network, Ipv4Mask  mask, Ipv4Address base, Ptr<NetDevice> device){
  Ipv4AddressHelper address;
  NetDeviceContainer devs(device);
  address.SetBase(network, mask, base);
  address.Assign(devs);
  return ;
}

uint32_t assign_addresses_to_devices(std::map<uint32_t, std::map<uint32_t, addr_entry_t> > &ADDR, NodeContainer & nodes){
  uint32_t addrCnt = 0;
  uint32_t nodeCnt = nodes.GetN();
  Ipv4AddressHelper address;
  for (uint32_t nodeIdx = 0; nodeIdx < nodeCnt; nodeIdx++) {
      Ptr<Node> curNode = nodes.Get(nodeIdx);
      std::map<uint32_t, std::map<uint32_t, addr_entry_t> >::iterator it_0 = ADDR.find(nodeIdx);
      if (it_0 == ADDR.end()) {
          std::cout<<"ERROR! assign_addresses_to_devices() with Node Index: " << nodeIdx << std::endl;
          return 0;
      }
      std::map<uint32_t, addr_entry_t> nodeADDR = it_0->second;
      uint32_t devCnt = curNode->GetNDevices();
      // std::cout << "Node "<< nodeIdx << " has " << devCnt-1 << " user-added devices and "<< nodeADDR.size() << " input addresses" << std::endl;
      for (uint32_t devIdx = 1; devIdx < devCnt; devIdx++) {
          Ptr<NetDevice> curDev = curNode->GetDevice(devIdx);
          if(curDev == 0){
            std::cout<<"ERROR! cannot get the device for node="<<nodeIdx<<" and deviceIdx="<<devIdx<<std::endl;
            return 0;
          }
          std::map<uint32_t, addr_entry_t> ::iterator it_1 = nodeADDR.find(devIdx);
          if (it_1 == nodeADDR.end()) {
              std::cout<<"ERROR! assign_addresses_to_devices() Error with Device Index: " << nodeIdx <<", "<< devIdx << std::endl;
              return 0;
          }
          assign_address_to_single_device(it_1->second.network, it_1->second.mask, it_1->second.base, curDev);
          addrCnt = addrCnt + 1;
      }
  }
  return addrCnt;
}

void record_addr_on_single_node(Ptr<Node> node, std::map<Ipv4Address, Ptr<Node> > & addr2node){
  auto ipv4 = node->GetObject<Ipv4>();
  auto nicCnt = ipv4->GetNInterfaces ();
  for (uint32_t i = 1; i < nicCnt; i++) {
    auto addr = ipv4->GetAddress(i, 0).GetLocal();
    addr2node[addr] = node;
  }
  return ;
}

void assign_addresses(NodeContainer & nodes, std::map<Ipv4Address, Ptr<Node> > & addr2node){
  uint32_t nodeCnt = nodes.GetN();
  for (uint32_t nodeIdx = 0; nodeIdx < nodeCnt; nodeIdx++) {
      Ptr<Node> curNode = nodes.Get(nodeIdx);
      assign_rdma_addresses_to_node(curNode);
      record_addr_on_single_node(curNode, addr2node);
  }
  return ;
}

NetDeviceContainer get_all_netdevices_of_a_node(Ptr<Node> node){
  NetDeviceContainer c;
  uint32_t devCnt = node->GetNDevices();
  for (uint32_t devIdx = 1; devIdx < devCnt; devIdx++) {
      Ptr<NetDevice> curDev = node->GetDevice(devIdx);
      c.Add(curDev);
  }
  return c;
}

void assign_rdma_addresses_to_node(Ptr<Node> node){
  // std::string msk = "255.255.255.0"
  Ipv4Mask msk("255.255.255.0");
  Ipv4Address base = string_to_ipv4Address("0.0.0.1");
  Ipv4Address ntwk(0x0b000000 + ((node->GetId() / 256) * 0x00010000) + ((node->GetId() % 256) * 0x00000100));
  NetDeviceContainer devs = get_all_netdevices_of_a_node(node);
  Ipv4AddressHelper hlpr;
  hlpr.SetBase(ntwk, msk, base);
  hlpr.Assign(devs);
  return ;
}



/* get average value of CDF distribution */
double avg_cdf(struct cdf_table *table) {
  int i = 0;
  double avg = 0;
  double value, prob;
  if (!table)
    return 0;
  for (i = 0; i < table->num_entry; i++) {
    if (i == 0) {
      value = table->entries[i].value / 2;
      prob = table->entries[i].cdf;
    }else {
      value = (table->entries[i].value + table->entries[i - 1].value) / 2;
      prob = table->entries[i].cdf - table->entries[i - 1].cdf;
    }
    avg += (value * prob);
  }
  return avg;
}

bool cmp_pitEntry_in_increase_order_of_latency(const PathData * lhs, const PathData * rhs) {
    return lhs->latency < rhs->latency; // 升序排列
}

bool cmp_pitEntry_in_decrease_order_of_priority(const PathData * lhs, const PathData * rhs) {
    return lhs->priority > rhs->priority; // 降序排列
}

bool cmp_pitEntry_in_increase_order_of_Generation_time(const PathData * lhs, const PathData * rhs){
    return lhs->tsGeneration < rhs->tsGeneration; // 升序排列
}

bool cmp_pitEntry_in_increase_order_of_congestion_Degree(const PathData * lhs, const PathData * rhs){
    return lhs->pathDre < rhs->pathDre; // 升序排列
}

bool cmp_pitEntry_in_increase_order_of_Sent_time(const PathData * lhs, const PathData * rhs){
    return lhs->tsLatencyLastSend < rhs->tsLatencyLastSend; // 升序排列
}


std::string construct_target_string(uint32_t strLen, std::string c) {
    std::string result;
    for (uint32_t i = 0; i < strLen; ++i) {
        result = result + c;
    }
    return result;
}

uint32_t create_topology( NodeContainer &switchNodes, NodeContainer &serverNodes, NodeContainer &allNodes, uint32_t switchNum, uint32_t serverNum) {
  switchNodes.Create(switchNum);
  serverNodes.Create(serverNum);
  allNodes.Add(switchNodes);
  allNodes.Add(serverNodes);
  return allNodes.GetN();
}

void free_cdf(struct cdf_table *table) {
  if (table)
    free(table->entries);
}

/* generate a random value based on CDF distribution */
double gen_random_cdf(struct cdf_table *table) {
  int i = 0;
  double x = rand_range(table->min_cdf, table->max_cdf);
  if (!table)
    return 0;
  for (i = 0; i < table->num_entry; i++)  {
    if (x <= table->entries[i].cdf) {
      if (i == 0)
        return interpolate(x, 0, 0, table->entries[i].cdf, table->entries[i].value);
      else
        return interpolate(x, table->entries[i - 1].cdf, table->entries[i - 1].value, table->entries[i].cdf, table->entries[i].value);
    }
  }
  return table->entries[table->num_entry - 1].value;
}

uint32_t hashing_flow_with_5_tuple(Ptr<const Packet> packet, const Ipv4Header &header){
    uint32_t flowId = 0;
    Hasher m_hasher;
    m_hasher.clear();
    TcpHeader tcpHeader;
    packet->PeekHeader(tcpHeader);
    std::ostringstream oss;
    oss << header.GetSource() << " ";
    oss << header.GetDestination() << " ";
    oss << header.GetProtocol() << " ";
    oss << tcpHeader.GetSourcePort() << " ";
    oss << tcpHeader.GetDestinationPort();
    std::string data = oss.str();
    flowId = m_hasher.GetHash32(data);
    return flowId;

    // uint32_t flowId = 0;
    // TcpHeader tcpHeader;
    // packet->PeekHeader(tcpHeader);

    // // 这是一个更简化的TCP头，我们需要手动解析
    // Ipv4Address srcAddr, dstAddr;
    // srcAddr = header.GetSource();
    // dstAddr = header.GetDestination();
    // uint8_t protocol = header.GetProtocol();
    // uint16_t sourcePort, destinationPort;
    // if(packet->GetSize() >= 12) { // minimal ack packet
    //   uint8_t buffer[12];
    //   packet->CopyData(buffer, 12);
    //   sourcePort = (buffer[0] << 8) + buffer[1];
    //   destinationPort = (buffer[2] << 8) + buffer[3];
    // }else{
    //   std::cout << "Error in packet size with TCP header without IP header! size = " <<  packet->GetSize() << std::endl;
    // }

    // Hasher m_hasher;
    // m_hasher.clear();
    // std::cout << "Header INFO: " << std::endl;
    // NS_LOG_INFO("header.GetSource(): " << srcAddr);
    // NS_LOG_INFO("header.GetDestination(): " << dstAddr);
    // NS_LOG_INFO("header.GetProtocol(): " << protocol);
    // NS_LOG_INFO("tcpHeader.GetDestinationPort(): " << sourcePort);
    // NS_LOG_INFO("tcpHeader.GetSourcePort(): " << destinationPort);

    // std::ostringstream oss;
    // oss << srcAddr << dstAddr << header.GetSource() << protocol << sourcePort << destinationPort;//获取源ip、目的ip、传输层协议
    // std::string data = oss.str();
    // flowId = m_hasher.GetHash32(data);
    // return flowId;
}

std::string hashing_flow_with_5_tuple_to_string(Ptr<const Packet> packet, const Ipv4Header &header){
    Hasher m_hasher;
    m_hasher.clear();
    TcpHeader tcpHeader;
    packet->PeekHeader(tcpHeader);
    std::ostringstream oss;
    oss << header.GetSource() << " ";
    oss << header.GetDestination() << " ";
    oss << header.GetProtocol() << " ";
    oss << tcpHeader.GetSourcePort() << " ";
    oss << tcpHeader.GetDestinationPort();
    std::string data = oss.str();
    return data;
}

void init_cdf(struct cdf_table *table) {
  if (!table)
    return;
  table->entries = (struct cdf_entry *)malloc(TG_CDF_TABLE_ENTRY * sizeof(struct cdf_entry));
  table->num_entry = 0;
  table->max_entry = TG_CDF_TABLE_ENTRY;
  table->min_cdf = 0;
  table->max_cdf = 1;
  if (!(table->entries))
    perror("Error: malloc entries in init_cdf()");
}

void install_flow_in_tcp_bulk_on_node_pair(Ptr<Node> srcServerNode, Ptr<Node> dstServerNode, uint16_t port, uint32_t flowSize,
                                   uint32_t packetSize, double startTime, double endTime){
      Ptr<Ipv4> ipv4 = dstServerNode->GetObject<Ipv4>();
      Ipv4InterfaceAddress dstInterface = ipv4->GetAddress(1, 0);
      Ipv4Address dstAddress = dstInterface.GetLocal();
      BulkSendHelper source("ns3::TcpSocketFactory", InetSocketAddress(dstAddress, port));
      source.SetAttribute("MaxBytes", UintegerValue(flowSize));
      source.SetAttribute ("SendSize", UintegerValue (packetSize));
      ApplicationContainer sourceApp = source.Install(srcServerNode);
      sourceApp.Start(Seconds(startTime));
      sourceApp.Stop(Seconds(endTime));
      PacketSinkHelper sink("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
      ApplicationContainer sinkApp = sink.Install(dstServerNode);
      sinkApp.Start(Seconds(startTime));
      sinkApp.Stop(Seconds(endTime));
}

void install_flows_in_tcp_bulk_on_node_pair(Ptr<Node> srcServerNode, Ptr<Node> dstServerNode,
                          double requestRate, struct cdf_table *cdfTable, long &flowCount, long &totalFlowSize,
                          double START_TIME, double END_TIME, double FLOW_LAUNCH_END_TIME,
                          uint16_t &appPort, uint32_t &smallFlowCount, uint32_t &largeFlowCount) {
  double startTime = START_TIME + poission_gen_interval(requestRate); // possion distribution of start time
  while (startTime < FLOW_LAUNCH_END_TIME) {
    // std::cout << "startTime :" << startTime << ", FLOW_LAUNCH_END_TIME : " << FLOW_LAUNCH_END_TIME << ", END_TIME : " << END_TIME;
    uint32_t flowSize = gen_random_cdf(cdfTable);
    install_tcp_bulk_on_node_pair(srcServerNode, dstServerNode, appPort, flowSize, startTime, END_TIME);
    startTime += poission_gen_interval(requestRate);
    appPort = appPort + 1;
    totalFlowSize += flowSize;
    flowCount = flowCount + 1;
    if (flowSize <= THRESHOLD_IN_BYTE_FOR_SMALL_FLOW) {
      smallFlowCount++;
    }else if (flowSize > THRESHOLD_IN_BYTE_FOR_LARGE_FLOW) {
      largeFlowCount++;
    }
  }
}

void addTcpSocketBaseBxCb(Ptr<PacketSink> sink, std::string fid, std::map<std::string, reorderDistEntry> * reorderDistTbl) {
      NS_LOG_FUNCTION ("addTcpSocketBaseBxCb()");
      if (sink == 0) {
        std::cout << "Error with null packet sink pointer in addTcpSocketBaseBxCb()" << std::endl;
        return ;
      }
      Ptr<Socket> skt = sink->GetListeningSocket();
      Ptr<TcpSocketBase> skt2 = DynamicCast<TcpSocketBase> (skt);     
      if (skt2 == 0) {
        std::cout << "Error with null packet TcpSocketBase pointer in addTcpSocketBaseBxCb()" << std::endl;
        return ;
      }
      NS_LOG_LOGIC ("At time " << Now() << ", Adding trace function for reorder statistics at receiver end for flow " << fid);
      skt2->TraceConnectWithoutContext ("Bx", MakeBoundCallback (&tcpSocketBaseBxCb, fid, reorderDistTbl));
      // Config::ConnectWithoutContext ("/NodeList/*/$ns3::TcpL4Protocol/SocketList/*/Bx", MakeBoundCallback (&tcpSocketBaseBxCb, reorderDistTbl));
}


void install_tcp_bulk_on_node_pair(Ptr<Node> srcServerNode, Ptr<Node> dstServerNode, uint16_t port, uint32_t flowSize, double START_TIME, double END_TIME){
      Ptr<Ipv4> ipv4 = dstServerNode->GetObject<Ipv4>();
      Ipv4InterfaceAddress dstInterface = ipv4->GetAddress(1, 0);
      Ipv4Address dstAddress = dstInterface.GetLocal();
      BulkSendHelper source("ns3::TcpSocketFactory", InetSocketAddress(dstAddress, port));
      source.SetAttribute("MaxBytes", UintegerValue(flowSize));
      ApplicationContainer sourceApp = source.Install(srcServerNode);
      sourceApp.Start(Seconds(START_TIME));
      sourceApp.Stop(Seconds(END_TIME));
      std::string flowId = std::to_string(srcServerNode->GetId()) + "-" + std::to_string(dstServerNode->GetId()) + "-" + std::to_string(port);
      sourceApp.Get(0)->TraceConnect ("AppStart", flowId, MakeCallback (&BulkSendApplication::StartNotification, DynamicCast<BulkSendApplication>(sourceApp.Get(0))));
      sourceApp.Get(0)->TraceConnect ("AppComplete", flowId, MakeCallback (&BulkSendApplication::CompleteNotification, DynamicCast<BulkSendApplication>(sourceApp.Get(0))));

      PacketSinkHelper sink("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
      ApplicationContainer sinkApp = sink.Install(dstServerNode);
      sinkApp.Start(Seconds(START_TIME));
      sinkApp.Stop(Seconds(END_TIME));

      Ptr<PacketSink> sinkPtr = DynamicCast<PacketSink> (sinkApp.Get (0));
      Simulator::Schedule(Seconds(START_TIME+0.000000001), &addTcpSocketBaseBxCb, sinkPtr, flowId, &BulkSendApplication::reorderDistTbl);

      // Ptr<PacketSink> sink = DynamicCast<PacketSink> (sinkApp.Get (0));
      // source->TraceConnectWithoutContext ("Tx", MakeCallback (&BulkSendBasicTestCase::SendTx, this));

      return ;
}

void install_rdma_client_on_single_node(Ptr<Node> node, flow_entry_t & f){
      RdmaClientHelper rdmaClientHelper(f.prioGroup, f.srcAddr, f.srcAddr, f.srcPort, f.dstPort, f.byteCnt, f.winInByte, f.rttInNs);
      ApplicationContainer rdmaClient = rdmaClientHelper.Install(node);
      rdmaClient.Start(NanoSeconds(f.startTimeInSec));
      return ;
}

void qp_finish(FILE * os, global_variable_t * m, Ptr<RdmaQueuePair> q){
    // sip, dip, sport, dport, dataSize (B), trafficSize,
    // start_time, last_time, cur_time, fct (ns), standalone_fct (ns)
    auto srcIpAddr =  q->sip;
    auto srcNode = m->addr2node[srcIpAddr];
    auto dstIpAddr =  q->dip;
    auto dstNode = m->addr2node[dstIpAddr];

    uint64_t baseRttInNs = m->pairRttInNs[srcNode][dstNode];
    uint64_t bitWdithPerSec = m->pairBwInBitps[srcNode][dstNode];
    uint32_t totalBytes = q->m_size + ((q->m_size-1) / m->defaultPktSizeInByte + 1) * (CustomHeader::GetStaticWholeHeaderSize() - IntHeader::GetStaticSize()); // translate to the minimum bytes required (with header but no INT)
    uint64_t baseFctInNs = baseRttInNs + totalBytes * 8000000000lu / bitWdithPerSec;

    fprintf(os, "SIP:%08x DIP:%08x SP:%u DP:%u DataSizeInByte:%lu PktSizeInByte:%u StartTimeInNs:%lu LastTimeInNs%lu EndTimeInNs:%lu BaseFctInNs:%lu\n",
                q->sip.Get(),
                q->dip.Get(),
                q->sport,
                q->dport,
                q->m_size,
                totalBytes,
                q->startTime.GetNanoSeconds(),
                (Simulator::Now() - q->startTime).GetNanoSeconds(),
                Simulator::Now().GetNanoSeconds(),
                baseFctInNs);
    // remove rxQp from the receiver
    Ptr<RdmaDriver> rdma = dstNode->GetObject<RdmaDriver> ();
    rdma->m_rdma->DeleteRxQp(q->sip.Get(), q->m_pg, q->sport);
}

void iterate_single_incast_kv_cache_application(global_variable_t * varMap, uint32_t jobIdx) {

    varMap->kvCachePara[jobIdx].roundCnt += 1;
    if (varMap->kvCachePara[jobIdx].roundCnt > varMap->kvCachePara[jobIdx].roundNum) {
      std::cout << "Successfully finish the " << jobIdx << " Job (Incast)" << std::endl;
      varMap->numOfFinishedJob += 1;

      return ;
    }
    NS_LOG_INFO("INCAST, JobID: " << jobIdx << ", Round **" << varMap->kvCachePara[jobIdx].roundCnt << "** Starts ");

    varMap->kvCachePara[jobIdx].completeCnt = 0;
    Ptr<Node> leaderNode =  varMap->kvCachePara[jobIdx].leaderNode;
    Ipv4Address leaderAddr = leaderNode->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();
    uint64_t delayInNs, flowSizeInByte;
    uint16_t appPort;
    if (varMap->kvCachePara[jobIdx].roundCnt != 1) {
      delayInNs = varMap->kvCachePara[jobIdx].otherTimeInNs + varMap->kvCachePara[jobIdx].reduceTimeInNs + varMap->kvCachePara[jobIdx].attentionTimeInNs;
    }else{
      delayInNs = 0;
    }
    for (uint32_t i = 0; i < varMap->kvCachePara[jobIdx].followerNodes.GetN(); i++) {
      Ptr<Node> followerNode =  varMap->kvCachePara[jobIdx].followerNodes.Get(i);
      Ipv4Address followerAddr = followerNode->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();
      varMap->appStartPort += 1;
      appPort = varMap->appStartPort;
      varMap->appPort2kvApp[appPort] = &varMap->kvCachePara[jobIdx];
      flowSizeInByte = varMap->kvCachePara[jobIdx].notifySizeInByte;
      RdmaClientHelper clientHelper(3, leaderAddr, followerAddr, appPort, appPort, flowSizeInByte, varMap->maxBdpInByte, varMap->maxRttInNs);
      ApplicationContainer appCon = clientHelper.Install(leaderNode);
      appCon.Start(NanoSeconds(delayInNs));
      NS_LOG_INFO("QP Pair Index: " << i << ", " <<
                  "Leader-->Follower: Job Index: " << jobIdx << ", " <<
                  "Type: " << varMap->kvCachePara[jobIdx].type << ", " <<
                  "Round: " << varMap->kvCachePara[jobIdx].roundCnt << ", " <<
                  "CompleteCnt: " << varMap->kvCachePara[jobIdx].completeCnt << ", " <<
                  "srcNode: " << leaderNode->GetId() << ", " <<
                  "DstNode: " << followerNode->GetId() << ", " <<
                  "appPort: " << appPort << ", " <<
                  "flowSizeInByte: " << flowSizeInByte << ", " <<
                  "WaitTimeInNs: " <<  delayInNs);

    }
}

void iterate_single_broadcast_kv_cache_application(global_variable_t * varMap, uint32_t jobIdx) {

    varMap->kvCachePara[jobIdx].roundCnt += 1;
    if (varMap->kvCachePara[jobIdx].roundCnt > varMap->kvCachePara[jobIdx].roundNum) {
      std::cout << "Successfully finish the " << jobIdx << " Job (Broadcast)" << std::endl;
      varMap->numOfFinishedJob += 1;
      return ;
    }
    NS_LOG_INFO("BROADCAST, JobID: " << jobIdx << ", Round **" << varMap->kvCachePara[jobIdx].roundCnt << "** Starts ");

    varMap->kvCachePara[jobIdx].completeCnt = 0;
    Ptr<Node> leaderNode =  varMap->kvCachePara[jobIdx].leaderNode;
    Ipv4Address leaderAddr = leaderNode->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();

    uint64_t delayInNs, flowSizeInByte;
    uint16_t appPort;
    if (varMap->kvCachePara[jobIdx].roundCnt != 1) {
      delayInNs = varMap->kvCachePara[jobIdx].reduceTimeInNs + varMap->kvCachePara[jobIdx].otherTimeInNs;
    }else{
      delayInNs = 0;
    }

    for (uint32_t i = 0; i < varMap->kvCachePara[jobIdx].followerNodes.GetN(); i++) {
      Ptr<Node> followerNode =  varMap->kvCachePara[jobIdx].followerNodes.Get(i);
      Ipv4Address followerAddr = followerNode->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();
      varMap->appStartPort += 1;
      appPort = varMap->appStartPort;
      varMap->appPort2kvApp[appPort] = &varMap->kvCachePara[jobIdx];
      flowSizeInByte = varMap->kvCachePara[jobIdx].querySizeInByte;
      RdmaClientHelper clientHelper(3, leaderAddr, followerAddr, appPort, appPort, flowSizeInByte, varMap->maxBdpInByte, varMap->maxRttInNs);
      ApplicationContainer appCon = clientHelper.Install(leaderNode);
      appCon.Start(NanoSeconds(delayInNs));


      NS_LOG_INFO("QP Pair Index: " << i << ", " <<
                  "Leader-->Follower: Job Index: " << jobIdx << ", " <<
                  "Type: " << varMap->kvCachePara[jobIdx].type << ", " <<
                  "Round: " << varMap->kvCachePara[jobIdx].roundCnt << ", " <<
                  "CompleteCnt: " << varMap->kvCachePara[jobIdx].completeCnt << ", " <<
                  "srcNode: " << leaderNode->GetId() << ", " <<
                  "DstNode: " << followerNode->GetId() << ", " <<
                  "appPort: " << appPort << ", " <<
                  "flowSizeInByte: " << flowSizeInByte << ", " <<
                  "WaitTimeInNs: " <<  delayInNs);

    }
}


void iterate_single_ring_kv_cache_application(global_variable_t * varMap, uint32_t jobIdx) {

    varMap->kvCachePara[jobIdx].roundCnt += 1;
    if (varMap->kvCachePara[jobIdx].roundCnt > varMap->kvCachePara[jobIdx].roundNum) {
      std::cout << "Successfully finish the " << jobIdx << " Job (Ringcast)" << std::endl;
      varMap->numOfFinishedJob += 1;
      return ;
    }
    NS_LOG_INFO("Start the **" << varMap->kvCachePara[jobIdx].roundCnt << "** round of the **" << jobIdx << "** Job (Ring)");

    varMap->kvCachePara[jobIdx].completeCnt = 0;


    uint64_t delayInNs, flowSizeInByte;
    uint16_t appPort;
    if (varMap->kvCachePara[jobIdx].roundCnt != 1) {
      delayInNs = varMap->kvCachePara[jobIdx].reduceTimeInNs + varMap->kvCachePara[jobIdx].otherTimeInNs;
    }else{
      delayInNs = 0;
    }

    for (uint32_t i = 0; i < varMap->kvCachePara[jobIdx].followerNodes.GetN(); i++) {
      Ptr<Node> followerNode =  varMap->kvCachePara[jobIdx].followerNodes.Get(i);
      Ipv4Address followerAddr = followerNode->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();
      Ptr<Node> leaderNode =  varMap->kvCachePara[jobIdx].leaderNodes.Get(i);
      Ipv4Address leaderAddr = leaderNode->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();
      varMap->appStartPort += 1;
      appPort = varMap->appStartPort;
      varMap->appPort2kvApp[appPort] = &varMap->kvCachePara[jobIdx];
      flowSizeInByte = varMap->kvCachePara[jobIdx].attentionSizeInByte;
      RdmaClientHelper clientHelper(3, leaderAddr, followerAddr, appPort, appPort, flowSizeInByte, varMap->maxBdpInByte, varMap->maxRttInNs);
      ApplicationContainer appCon = clientHelper.Install(leaderNode);
      appCon.Start(NanoSeconds(delayInNs));


      NS_LOG_INFO("QP Pair Index: " << i << ", " <<
                  "Leader-->Follower: Job Index: " << jobIdx << ", " <<
                  "Type: " << varMap->kvCachePara[jobIdx].type << ", " <<
                  "Round: " << varMap->kvCachePara[jobIdx].roundCnt << ", " <<
                  "CompleteCnt: " << varMap->kvCachePara[jobIdx].completeCnt << ", " <<
                  "srcNode: " << leaderNode->GetId() << ", " <<
                  "DstNode: " << followerNode->GetId() << ", " <<
                  "appPort: " << appPort << ", " <<
                  "flowSizeInByte: " << flowSizeInByte << ", " <<
                  "WaitTimeInNs: " <<  delayInNs);

    }
}


void iterate_single_inca_kv_cache_application(global_variable_t * varMap, uint32_t jobIdx) {

    varMap->kvCachePara[jobIdx].roundCnt += 1;
    varMap->kvCachePara[jobIdx].completeCnt = 0;
    if (varMap->kvCachePara[jobIdx].roundCnt > varMap->kvCachePara[jobIdx].roundNum) {
      std::cout << "Successfully finish the " << jobIdx << " Job (INCA)" << std::endl;
      varMap->numOfFinishedJob += 1;
      return ;
    }
    NS_LOG_INFO("Start the **" << varMap->kvCachePara[jobIdx].roundCnt << "** round of the **" << jobIdx << "** Job (Inca)");

    uint64_t delayInNs, flowSizeInByte = varMap->kvCachePara[jobIdx].querySizeInByte;
    uint16_t appPort;
    if (varMap->kvCachePara[jobIdx].roundCnt != 1) {
      delayInNs = varMap->kvCachePara[jobIdx].otherTimeInNs;
    }else{
      delayInNs = 0;
    }

    for (uint32_t i = 0; i < varMap->kvCachePara[jobIdx].followerNodes.GetN(); i++) {
      Ptr<Node> leaderNode =  varMap->kvCachePara[jobIdx].leaderNodes.Get(i);
      Ipv4Address leaderAddr = leaderNode->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();
      Ptr<Node> followerNode =  varMap->kvCachePara[jobIdx].followerNodes.Get(i);
      Ipv4Address followerAddr = followerNode->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();
      varMap->appStartPort += 1;
      appPort = varMap->appStartPort;
      varMap->appPort2kvApp[appPort] = &varMap->kvCachePara[jobIdx];
      RdmaClientHelper clientHelper(3, leaderAddr, followerAddr, appPort, appPort, flowSizeInByte, varMap->maxBdpInByte, varMap->maxRttInNs);
      ApplicationContainer appCon = clientHelper.Install(leaderNode);
      appCon.Start(NanoSeconds(delayInNs));


      NS_LOG_INFO("QP Pair Index: " << i << ", " <<
                  "Leader-->Follower: Job Index: " << jobIdx << ", " <<
                  "Type: " << varMap->kvCachePara[jobIdx].type << ", " <<
                  "Round: " << varMap->kvCachePara[jobIdx].roundCnt << ", " <<
                  "CompleteCnt: " << varMap->kvCachePara[jobIdx].completeCnt << ", " <<
                  "srcNode: " << leaderNode->GetId() << ", " <<
                  "DstNode: " << followerNode->GetId() << ", " <<
                  "appPort: " << appPort << ", " <<
                  "flowSizeInByte: " << flowSizeInByte << ", " <<
                  "WaitTimeInNs: " <<  delayInNs);

    }
}



void qp_finish_kv_cache(FILE * os, global_variable_t * m, Ptr<RdmaQueuePair> q){
    // sip, dip, sport, dport, dataSize (B), trafficSize,
    // start_time, last_time, cur_time, fct (ns), standalone_fct (ns)

    kv_cache_para_t* kvApp = m->appPort2kvApp[q->sport];
    uint32_t jobIdx = kvApp->idx;
    m->appPort2kvApp.erase(q->sport);

    auto srcIpAddr =  q->sip;
    auto srcNode = m->addr2node[srcIpAddr];
    auto dstIpAddr =  q->dip;
    auto dstNode = m->addr2node[dstIpAddr];
    // uint64_t baseRttInNs = m->pairRttInNs[srcNode][dstNode];
    // uint64_t bitWdithPerSec = m->pairBwInBitps[srcNode][dstNode];
    uint32_t totalBytes = q->m_size + ((q->m_size-1) / m->defaultPktSizeInByte + 1) * (CustomHeader::GetStaticWholeHeaderSize() - IntHeader::GetStaticSize()); // translate to the minimum bytes required (with header but no INT)
    // uint64_t baseFctInNs = baseRttInNs + totalBytes * 8000000000lu / bitWdithPerSec;
    // fprintf(os, "Job:%d Round:%d Complete:%d srcNode:%d dstNode:%d DataSizeInByte:%lu PktSizeInByte:%u StartTimeInNs:%lu LastTimeInNs:%lu EndTimeInNs:%lu\n",
    //             jobIdx, kvApp->roundCnt, kvApp->completeCnt, srcNode->GetId(), dstNode->GetId(), q->m_size, totalBytes, q->startTime.GetNanoSeconds(),
    //             (Simulator::Now() - q->startTime).GetNanoSeconds(), Simulator::Now().GetNanoSeconds());



    if (kvApp->type == KV_CACHE_INCAST) {
      if (srcNode == kvApp->leaderNode) { // leader --> follower
        m->appStartPort += 1;
        uint16_t appPort = m->appStartPort;
        m->appPort2kvApp[appPort] = kvApp;
        uint64_t flowSizeInByte = kvApp->attentionSizeInByte;
        uint64_t delayInNs = 0;
        RdmaClientHelper clientHelper(3, dstIpAddr, srcIpAddr, appPort, appPort, flowSizeInByte, m->maxBdpInByte, m->maxRttInNs);
        ApplicationContainer appCon = clientHelper.Install(dstNode);
        appCon.Start(NanoSeconds(delayInNs));
        NS_LOG_INFO("Start Follower-->Leader, Job Index: " << jobIdx << ", " <<
                    "Type: " <<kvApp->type << ", " <<
                    "Round: " << kvApp->roundCnt << ", " <<
                    "CompleteCnt: " << kvApp->completeCnt << ", " <<
                    "srcNode: " << dstNode->GetId() << ", " <<
                    "DstNode: " << srcNode->GetId() << ", " <<
                    "appPort: " << appPort << ", " <<
                    "flowSizeInByte: " << flowSizeInByte << ", " <<
                    "WaitTimeInNs: " <<  delayInNs);
      }else{
        kvApp->completeCnt += 1;
        fprintf(os, "Job:%d Round:%d Complete:%d srcNode:%d dstNode:%d DataSizeInByte:%lu PktSizeInByte:%u StartTimeInNs:%lu LastTimeInNs:%lu EndTimeInNs:%lu\n",
                    jobIdx, kvApp->roundCnt, kvApp->completeCnt, srcNode->GetId(), dstNode->GetId(), q->m_size, totalBytes, q->startTime.GetNanoSeconds(),
                    (Simulator::Now() - q->startTime).GetNanoSeconds(), Simulator::Now().GetNanoSeconds());
        fflush(os);
        if (kvApp->completeCnt == kvApp->completeNum) {
            iterate_single_incast_kv_cache_application(m, jobIdx);
        }

      }
    }else if (kvApp->type == KV_CACHE_BROADCAST) {
      if (srcNode == kvApp->leaderNode) { // leader --> follower
        m->appStartPort += 1;
        uint16_t appPort = m->appStartPort;
        m->appPort2kvApp[appPort] = kvApp;
        uint64_t flowSizeInByte = kvApp->querySizeInByte;
        uint64_t delayInNs = kvApp->attentionTimeInNs;
        RdmaClientHelper clientHelper(3, dstIpAddr, srcIpAddr, appPort, appPort, flowSizeInByte, m->maxBdpInByte, m->maxRttInNs);
        ApplicationContainer appCon = clientHelper.Install(dstNode);
        appCon.Start(NanoSeconds(delayInNs));
        NS_LOG_INFO("Start Follower-->Leader, Job Index: " << jobIdx << ", " <<
                    "Type: " <<kvApp->type << ", " <<
                    "Round: " << kvApp->roundCnt << ", " <<
                    "CompleteCnt: " << kvApp->completeCnt << ", " <<
                    "srcNode: " << dstNode->GetId() << ", " <<
                    "DstNode: " << srcNode->GetId() << ", " <<
                    "appPort: " << appPort << ", " <<
                    "flowSizeInByte: " << flowSizeInByte << ", " <<
                    "WaitTimeInNs: " <<  delayInNs);
      }else{
        kvApp->completeCnt += 1;
        fprintf(os, "Job:%d Round:%d Complete:%d srcNode:%d dstNode:%d DataSizeInByte:%lu PktSizeInByte:%u StartTimeInNs:%lu LastTimeInNs:%lu EndTimeInNs:%lu\n",
                    jobIdx, kvApp->roundCnt, kvApp->completeCnt, srcNode->GetId(), dstNode->GetId(), q->m_size, totalBytes, q->startTime.GetNanoSeconds(),
                    (Simulator::Now() - q->startTime).GetNanoSeconds(), Simulator::Now().GetNanoSeconds());
        fflush(os);
        if (kvApp->completeCnt == kvApp->completeNum) {
            iterate_single_broadcast_kv_cache_application(m, jobIdx);
        }
      }
    }else if (kvApp->type == KV_CACHE_INCA) {
      if (kvApp->nodeTypeMap[srcNode] == "LEADER") { // leader --> follower
        m->appStartPort += 1;
        uint16_t appPort = m->appStartPort;
        m->appPort2kvApp[appPort] = kvApp;
        uint64_t flowSizeInByte = kvApp->querySizeInByte;
        uint64_t delayInNs = kvApp->attentionTimeInNs;
        RdmaClientHelper clientHelper(3, dstIpAddr, srcIpAddr, appPort, appPort, flowSizeInByte, m->maxBdpInByte, m->maxRttInNs);
        ApplicationContainer appCon = clientHelper.Install(dstNode);
        appCon.Start(NanoSeconds(delayInNs));
        NS_LOG_INFO("Start Follower-->Leader, Job Index: " << jobIdx << ", " <<
                    "Type: " <<kvApp->type << ", " <<
                    "Round: " << kvApp->roundCnt << ", " <<
                    "CompleteCnt: " << kvApp->completeCnt << ", " <<
                    "srcNode: " << dstNode->GetId() << ", " <<
                    "DstNode: " << srcNode->GetId() << ", " <<
                    "appPort: " << appPort << ", " <<
                    "flowSizeInByte: " << flowSizeInByte << ", " <<
                    "WaitTimeInNs: " <<  delayInNs);
      }else{
        kvApp->completeCnt += 1;
        fprintf(os, "Job:%d Round:%d Complete:%d srcNode:%d dstNode:%d DataSizeInByte:%lu PktSizeInByte:%u StartTimeInNs:%lu LastTimeInNs:%lu EndTimeInNs:%lu\n",
                    jobIdx, kvApp->roundCnt, kvApp->completeCnt, srcNode->GetId(), dstNode->GetId(), q->m_size, totalBytes, q->startTime.GetNanoSeconds(),
                    (Simulator::Now() - q->startTime).GetNanoSeconds(), Simulator::Now().GetNanoSeconds());
        fflush(os);

        if (kvApp->completeCnt == kvApp->completeNum) {
            iterate_single_inca_kv_cache_application(m, jobIdx);
        }
      }
    }else if (kvApp->type == KV_CACHE_RING) {
        kvApp->completeCnt += 1;
        fprintf(os, "Job:%d Round:%d Complete:%d srcNode:%d dstNode:%d DataSizeInByte:%lu PktSizeInByte:%u StartTimeInNs:%lu LastTimeInNs:%lu EndTimeInNs:%lu\n",
                    jobIdx, kvApp->roundCnt, kvApp->completeCnt, srcNode->GetId(), dstNode->GetId(), q->m_size, totalBytes, q->startTime.GetNanoSeconds(),
                    (Simulator::Now() - q->startTime).GetNanoSeconds(), Simulator::Now().GetNanoSeconds());
        fflush(os);
        if (kvApp->completeCnt == kvApp->completeNum) {
            iterate_single_ring_kv_cache_application(m, jobIdx);
        }
      
    }else{
      std::cout << "ERROR Code 666" << std::endl;
    }
    
    // remove rxQp from the receiver
    Ptr<RdmaDriver> rdma = dstNode->GetObject<RdmaDriver> ();
    rdma->m_rdma->DeleteRxQp(q->dip.Get(), q->m_pg, q->dport);
}


void install_tcp_bulk_on_nodes( std::map<uint32_t, std::vector<tfc_entry_t> >& TFC,
                                    NodeContainer & nodes, struct cdf_table *cdfTable,
                                    double START_TIME, double END_TIME, double FLOW_LAUNCH_END_TIME,
                                    uint16_t startAppPort, double loadFactor,
                                    long &flowCount, uint32_t &smallFlowCount, uint32_t &largeFlowCount,
                                    long &totalFlowSize) {
  // source.SetAttribute ("SendSize", UintegerValue (PACKET_SIZE));
  std::map<uint32_t, std::vector<tfc_entry_t> >::iterator it_0;
  for (it_0 = TFC.begin(); it_0 != TFC.end(); it_0++) {
    Ptr<Node> srcNode = nodes.Get(it_0->first);
    std::vector<tfc_entry_t> tfcEntries = it_0->second;
    // std::cout << "srcNode ID: " <<  srcNode->GetId() << construct_target_string(5, " ");
    for (uint32_t i = 0; i < tfcEntries.size(); i++) {
      // std::cout << "sequence : " << i << construct_target_string(5, " ");
      tfc_entry_t tfcEntry = tfcEntries[i];
      uint32_t dstNodeIdx = tfcEntry.dstNodeIdx;
      Ptr<Node> dstNode = nodes.Get(dstNodeIdx);
      // std::cout << "dstNode ID: " <<  dstNode->GetId() << construct_target_string(5, " ");
      double loadfactorAdjustFacror = tfcEntry.loadfactor;
      // std::cout << "loadfactorAdjustFacror: " <<  loadfactorAdjustFacror << construct_target_string(5, " ");
      double tmpLoadFactor = loadfactorAdjustFacror * loadFactor;
      // std::cout << "tmpLoadFactor: " <<  tmpLoadFactor << construct_target_string(5, " ");
      uint32_t widthInGbps = tfcEntry.capacityInGbps;
      // std::cout << "widthInGbps: " <<  widthInGbps << construct_target_string(5, " ");
      double requestRate = tmpLoadFactor * widthInGbps * BYTE_NUMBER_PER_GBPS / (8 * avg_cdf (cdfTable));
      // std::cout << "requestRate: " <<  requestRate << construct_target_string(5, " ");
      long tmpTotalFlowSizeInByte = 0, tmpFlowCount = 0;
      uint32_t tmpSmallFlowCount=0, tmpLargeFlowCount=0;
      install_flows_in_tcp_bulk_on_node_pair( srcNode, dstNode, requestRate, cdfTable,
                                              tmpFlowCount, tmpTotalFlowSizeInByte,
                                              START_TIME, END_TIME, FLOW_LAUNCH_END_TIME,
                                              startAppPort, tmpSmallFlowCount, tmpLargeFlowCount);
      it_0->second[i].flowCount = tmpFlowCount;
      it_0->second[i].bytesCount = tmpTotalFlowSizeInByte;
      it_0->second[i].smallFlowCount = tmpSmallFlowCount;
      it_0->second[i].largeFlowCount = tmpLargeFlowCount;

      flowCount = flowCount + tmpFlowCount;
      totalFlowSize = totalFlowSize + tmpTotalFlowSizeInByte;
      smallFlowCount = smallFlowCount + tmpSmallFlowCount;
      largeFlowCount = largeFlowCount + tmpLargeFlowCount;
    }
  }
}


void install_tcp_test_applications (Ptr<Node>  srcNode, Ptr<Node>  dstNode, double START_TIME, double END_TIME,
                                    uint16_t port, const uint32_t packetCount, uint32_t intervalInNs) {
      NS_LOG_INFO("Node " << srcNode->GetId() << " sends "  << packetCount << " packets to Node " << dstNode->GetId());
      Ptr<Ipv4> ipv4 = dstNode->GetObject<Ipv4>();
      Ipv4InterfaceAddress dstInterface = ipv4->GetAddress(1, 0);
      Ipv4Address dstAddress = dstInterface.GetLocal();
      
      BulkSendHelper source("ns3::TcpSocketFactory", InetSocketAddress(dstAddress, port)); 
      source.SetAttribute ("SendSize", UintegerValue (DEFAULT_MAX_TCP_MSS_IN_BYTE));
      source.SetAttribute ("MaxBytes", UintegerValue (DEFAULT_MAX_TCP_MSS_IN_BYTE*packetCount));
      ApplicationContainer sourceApp = source.Install(srcNode);
      sourceApp.Start(Seconds(START_TIME));
      sourceApp.Stop(Seconds(END_TIME));
      std::string flowId = std::to_string(srcNode->GetId()) + "-" + std::to_string(dstNode->GetId()) + "-" + std::to_string(port);
      sourceApp.Get(0)->TraceConnect ("AppStart", flowId, MakeCallback (&BulkSendApplication::StartNotification, DynamicCast<BulkSendApplication>(sourceApp.Get(0))));
      sourceApp.Get(0)->TraceConnect ("AppComplete", flowId, MakeCallback (&BulkSendApplication::CompleteNotification, DynamicCast<BulkSendApplication>(sourceApp.Get(0))));

      PacketSinkHelper sink("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
      ApplicationContainer sinkApp = sink.Install(dstNode);
      sinkApp.Start(Seconds(START_TIME));
      sinkApp.Stop(Seconds(END_TIME));

      Ptr<PacketSink> sinkPtr = DynamicCast<PacketSink> (sinkApp.Get (0));
      Simulator::Schedule(Seconds(START_TIME+0.000000001), &addTcpSocketBaseBxCb, sinkPtr, flowId, &BulkSendApplication::reorderDistTbl);

      return ;
}



void install_tcp_test_applications_2 (Ptr<Node>  srcNode, Ptr<Node>  dstNode, double START_TIME, double END_TIME,
                                    uint16_t appPort, const uint32_t packetCount, uint32_t intervalInNs) {
      for (size_t i = 0; i < packetCount; i++) {
        std::cout<<"Node " << srcNode->GetId() << " sends "  << 1 << " packet to Node " << dstNode->GetId() << std::endl;
        uint16_t port = appPort++; // the dst app port
        Ptr<Ipv4> ipv4 = dstNode->GetObject<Ipv4>();
        Ipv4InterfaceAddress dstInterface = ipv4->GetAddress(1, 0);
        Ipv4Address dstAddress = dstInterface.GetLocal();
        
        BulkSendHelper source("ns3::TcpSocketFactory", InetSocketAddress(dstAddress, port)); 
        // source.SetAttribute("MaxBytes", UintegerValue(packetCount*ECHO_PKT_SIZE_IN_BYTE));
        source.SetAttribute ("SendSize", UintegerValue (DEFAULT_MAX_TCP_MSS_IN_BYTE));
        source.SetAttribute ("MaxBytes", UintegerValue (DEFAULT_MAX_TCP_MSS_IN_BYTE));
        // source.SetAttribute ("PacketSize", UintegerValue (ECHO_PKT_SIZE_IN_BYTE));
        ApplicationContainer sourceApp = source.Install(srcNode);
        sourceApp.Start(NanoSeconds(START_TIME+(i+1)*intervalInNs));
        sourceApp.Stop(Seconds(END_TIME));
        PacketSinkHelper sink("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
        ApplicationContainer sinkApp = sink.Install(dstNode);
        sinkApp.Start(Seconds(START_TIME));
        sinkApp.Stop(Seconds(END_TIME));
      }
      return ;
}

void install_udp_echo_applications (Ptr<Node>  srcNode, Ptr<Node>  dstNode, const uint32_t packetCount) {
  std::cout<<"Node " << srcNode->GetId() << " sends "  << packetCount << " packets to Node " << dstNode->GetId() << std::endl;

  UdpEchoClientHelper echoClient (dstNode->GetObject<Ipv4>()->GetAddress(1,0).GetLocal(), DEFAULT_UDP_ECHO_PORT_NUMBER);
  echoClient.SetAttribute ("MaxPackets", UintegerValue (packetCount));
  echoClient.SetAttribute ("Interval", TimeValue (NanoSeconds (DEFAULT_UDP_ECHO_PKT_INTERVAL_IN_NANOSECOND)));
  echoClient.SetAttribute ("PacketSize", UintegerValue (DEFAULT_UDP_ECHO_PKT_SIZE_IN_BYTE));
  ApplicationContainer clientApps = echoClient.Install (srcNode);
  clientApps.Start (Seconds (DEFAULT_UDP_ECHO_START_TIME_IN_SECOND));
  clientApps.Stop (Seconds (DEFAULT_UDP_ECHO_END_TIME_IN_SECOND));

  UdpEchoServerHelper echoServer(DEFAULT_UDP_ECHO_PKT_SIZE_IN_BYTE);
  ApplicationContainer serverApps = echoServer.Install (dstNode);
  serverApps.Start (Seconds (DEFAULT_UDP_ECHO_START_TIME_IN_SECOND));
  serverApps.Stop (Seconds (DEFAULT_UDP_ECHO_END_TIME_IN_SECOND));
  return ;
}


std::string integer_vector_to_string_with_range_merge(std::vector<uint32_t>& v) {
    if (v.empty()) {
        return "[NULL]";
    }

    std::string result = "[";
    size_t start = 0;
    size_t end = 0;

    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i] == v[i - 1] + 1) {
            end = i;
        } else {
            if (start == end) {
                result += to_string(v[start]) + ", ";
            } else {
                result += to_string(v[start]) + "~" + to_string(v[end]) + ", ";
            }
            start = i;
            end = i;
        }
    }

    if (start == end) {
        result += to_string(v[start]);
    } else {
        result += to_string(v[start]) + "~" + to_string(v[end]);
    }

    result += "]";
    return result;
}

double interpolate(double x, double x1, double y1, double x2, double y2) {
  if (x1 == x2)
    return (y1 + y2) / 2;
  else
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1);
}

std::string ipv4Address_to_string(Ipv4Address addr) {
  std::ostringstream os;
  addr.Print(os);
  std::string addrStr = os.str();
  return addrStr;
}

void load_cdf(struct cdf_table *table, const char *file_name){
  FILE *fd = NULL;
  char line[256] = {0};
  struct cdf_entry *e = NULL;
  int i = 0;
  if (!table)
    return;
  fd = fopen(file_name, "r");
  if (!fd)
    perror("Error: open the CDF file in load_cdf()");
  while (fgets(line, sizeof(line), fd)) {
    /* resize entries */
    if (table->num_entry >= table->max_entry) {
      table->max_entry *= 2;
      e = (struct cdf_entry *)malloc(table->max_entry * sizeof(struct cdf_entry));
      if (!e)
        perror("Error: malloc entries in load_cdf()");
      for (i = 0; i < table->num_entry; i++)
        e[i] = table->entries[i];
      free(table->entries);
      table->entries = e;
    }
    sscanf(line, "%lf %lf", &(table->entries[table->num_entry].value), &(table->entries[table->num_entry].cdf));
    if (table->min_cdf > table->entries[table->num_entry].cdf)
      table->min_cdf = table->entries[table->num_entry].cdf;
    if (table->max_cdf < table->entries[table->num_entry].cdf)
      table->max_cdf = table->entries[table->num_entry].cdf;
    table->num_entry++;
  }
  fclose(fd);
}

double poission_gen_interval(double avg_rate) {
  if (avg_rate > 0)
    return -logf(1.0 - (double)rand() / RAND_MAX) / avg_rate;
  else
    return 0;
}

uint32_t print_address_for_K_th_device(Ptr<Node> & curNode, uint32_t k){
  uint32_t curNodeIdx = curNode->GetId();
  Ptr<Ipv4> curIpv4 = curNode->GetObject<Ipv4> ();
  uint32_t intfCnt = curIpv4->GetNInterfaces ();
  if (k >= intfCnt) {
    std::cout << "Error in print_address_for_K_th_device() about curNodeIdx=" <<\
                 curNodeIdx << ", intfCnt=" << intfCnt << ", curIntfIdx=" << k << std::endl;
    return 0;
  }
  Ipv4InterfaceAddress curIpv4Addr = curIpv4->GetAddress (k,0);
  Ipv4Address curAddr = curIpv4Addr.GetLocal(); // Get the IPv4 address
  std::string curAddrStr = ipv4Address_to_string(curAddr);
  std::cout <<"The "<< k <<"-th user-added NIC of the total "<< intfCnt -1\
                    << " user-added NICs for the Node "<< curNodeIdx \
                    <<" is assigned with the address "<< curAddrStr << std::endl;
  return 1;
}

uint32_t print_address_for_single_node(Ptr<Node> & curNode){
  uint32_t nodeAddrCnt = 0;
  uint32_t curNodeIdx = curNode->GetId();
  Ptr<Ipv4> curIpv4 = curNode->GetObject<Ipv4> ();
  uint32_t intfCnt = curIpv4->GetNInterfaces ();
  std::cout << "Node " << curNodeIdx << " has " << intfCnt << " user-added NICs in total" << std::endl;
  for (uint32_t i = 1; i < intfCnt; i++) {
    nodeAddrCnt = nodeAddrCnt + print_address_for_K_th_device(curNode, i);
  }
  
  return nodeAddrCnt;
}

uint32_t print_addresses_for_nodes(NodeContainer &allNodes){
  uint32_t  totalAddrCnt = 0;
  uint32_t nodeCnt = allNodes.GetN();
  for (uint32_t i = 0; i < nodeCnt; i++) {
    Ptr<Node> curNode = allNodes.Get(i);
    uint32_t curAddrCnt =  print_address_for_single_node(curNode);
    totalAddrCnt = totalAddrCnt + curAddrCnt;
  }
  return totalAddrCnt;
}

void print_cdf(struct cdf_table *table) {
  int i = 0;
  if (!table)
    return;
  for (i = 0; i < table->num_entry; i++)
    printf("%.2f %.2f\n", table->entries[i].value, table->entries[i].cdf);
}

uint32_t print_connections_for_single_node(Ptr<Node> nodeA){
  uint32_t channelCnt = nodeA->GetNDevices ();
  std::cout << "Node "<< nodeA->GetId() <<" has "<< channelCnt-1 <<  " user-added connections" << std::endl;
  for (uint32_t j = 1; j < channelCnt; ++j) {
      Ptr<NetDevice> device = nodeA->GetDevice(j);
      Ptr<Channel> channel = device->GetChannel();
      for (uint32_t k = 0; k < channel->GetNDevices(); ++k) {
        Ptr<NetDevice> connected_device = channel->GetDevice(k);
        if (connected_device == device) {// Skip the device of node A
            continue;
        }
        uint32_t connected_device_idx = connected_device->GetIfIndex();
        Ptr<Node> connected_node = connected_device->GetNode();
        uint32_t connected_node_id = connected_node->GetId();
        std::cout <<"Node "<< nodeA->GetId() <<"'s "<< j << "-th NIC <-------> to Node "\
                  << connected_node_id << "'s " << connected_device_idx << "-th NIC"<< std::endl;
      }
    }
  return channelCnt;
}

uint32_t print_connections_for_nodes(NodeContainer &allNodes){
  NS_LOG_INFO("----------The Connections Between Nodes----------");
  uint32_t nodeCnt = allNodes.GetN();
  uint32_t totalChannelCnt = 0;
  for (uint32_t i = 0; i < nodeCnt; i++) {
    Ptr<Node> curNode = allNodes.Get(i);
    uint32_t curNodeChannelCnt = print_connections_for_single_node(curNode);
    totalChannelCnt = totalChannelCnt + curNodeChannelCnt;
  }
  return totalChannelCnt;
}

void print_EST_to_file(std::string outputFileName, std::map<uint32_t, est_entry_t> &est) {
  std::ofstream os(outputFileName.c_str());
  if (!os.is_open()) {
    std::cout << "print_EST_to_file() cannot open file " << outputFileName << std::endl;
    return ;
  }
  uint32_t estSize = est.size();
  os << "Index" << construct_target_string(5, " ");
  os << "Name" << construct_target_string(80, " ");
  os << "Value" << std::endl;
  
  for (uint32_t i=0; i < estSize; i++) {

    std::string name = est[i].name;
    std::string value = est[i].value;
    // std::cout << "i=" << i << ", name=" << name <<", value=" << value << std::endl;
    os << i << construct_target_string(5+5-to_string(i).size(), " ");
    os << name << construct_target_string(80+4-to_string(name).size(), " ");
    os << value << std::endl;
  }
  os.close();
}

void print_probe_info_to_file(std::string outputFileName, std::vector <probeInfoEntry> &m) {
  std::ofstream os(outputFileName.c_str());
  if (!os.is_open()){
    std::cout << "print_probe_info_to_file cannot open file " << outputFileName << std::endl;
  }
  uint32_t i = 0;
  for (auto & e : m ) {
    e.pathId = i;
    i++;
  }
  for (auto & e : m ) {
    os << e.pathId <<" "<< e.probeCnt << std::endl;
  }
  os.close();
}

void print_reorder_info_to_file(std::string outputFileName, std::map <std::string, reorder_entry_t> &m_reorderTable) {
  std::ofstream os(outputFileName.c_str());
  if (!os.is_open()){
    std::cout << "print_reorder_info_to_file() cannot open file " << outputFileName << std::endl;
  }
  uint32_t idx = 0;
  for (auto & it : m_reorderTable) {
    os << idx << ", ";
    // os << it.first << ", ";
    if (it.second.flag == false){
      os << "False" <<", ";
    }else{
      os << "True" <<", ";
    }
    os << vector_to_string<uint32_t>(it.second.seqs);    
    os << std::endl;
    idx += 1;
  }
  os.close();
}


void print_TFC(std::map<uint32_t, std::vector<tfc_entry_t> >& TFC, double timeGap){
  std::map<uint32_t, std::vector<tfc_entry_t> >::iterator it_0;
  std::cout << "Index" << construct_target_string(2, " ");
  std::cout << "SrcNode" << construct_target_string(2, " ");
  std::cout << "Loadfactor" << construct_target_string(2, " ");
  std::cout << "FlowCnt" << construct_target_string(2, " ");
  std::cout << "ByteCnt" << construct_target_string(5, " ");
  std::cout << "AvgFlowByte" << construct_target_string(5, " ");
  std::cout << "sFlowCnt" << construct_target_string(2, " ");
  std::cout << "lFlowCnt" << construct_target_string(2, " ");
  std::cout << "DstNodes" << construct_target_string(2, " ");
  std::cout << std::endl;

  uint32_t index = 0;
  long totalBytesCount = 0;
  uint32_t capacityInGbps = 0;

  for (it_0 = TFC.begin(); it_0 != TFC.end(); it_0++) {
      uint32_t srcNodeIdx = it_0->first;
      long flowCount = 0;
      long bytesCount = 0;
      uint32_t smallFlowCount = 0;
      uint32_t largeFlowCount = 0;
      std::vector<uint32_t> dstNodesIdx;
      std::vector<tfc_entry_t> tfcEntries = it_0->second;
      for (uint32_t i = 0; i < tfcEntries.size(); i++) {
        dstNodesIdx.push_back(tfcEntries[i].dstNodeIdx);
        flowCount = flowCount + tfcEntries[i].flowCount;
        bytesCount = bytesCount + tfcEntries[i].bytesCount;
        smallFlowCount = smallFlowCount + tfcEntries[i].smallFlowCount;
        largeFlowCount = largeFlowCount + tfcEntries[i].largeFlowCount;
        capacityInGbps = tfcEntries[i].capacityInGbps;
      }
      totalBytesCount = totalBytesCount + bytesCount;
      double AvgFlowSize = 0;
      if (flowCount != 0){
        AvgFlowSize = 1.0 * bytesCount / flowCount;
      }
       
      double lf = 1.0 * bytesCount / timeGap / capacityInGbps / BYTE_NUMBER_PER_GBPS * 100 * 8;
      std::cout << index << construct_target_string(5+2-to_string(index).size(), " ");
      std::cout << srcNodeIdx << construct_target_string(7+2-to_string(srcNodeIdx).size(), " ");
      std::cout << to_string(lf) << construct_target_string(10+2-to_string(lf).size(), " ");
      std::cout << flowCount << construct_target_string(7+2-to_string(flowCount).size(), " ");
      std::cout << bytesCount << construct_target_string(7+5-to_string(bytesCount).size(), " ");
      std::cout << to_string(AvgFlowSize) << construct_target_string(11+5-to_string(AvgFlowSize).size(), " ");
      std::cout << smallFlowCount << construct_target_string(8+2-to_string(smallFlowCount).size(), " ");
      std::cout << largeFlowCount << construct_target_string(8+2-to_string(largeFlowCount).size(), " ");
      std::cout << integer_vector_to_string_with_range_merge(dstNodesIdx);
      std::cout << std::endl;
      index = index + 1;
  }
  double totalLf = 1.0 * totalBytesCount / timeGap / capacityInGbps / BYTE_NUMBER_PER_GBPS * 100 * 8 / TFC.size();
  std::cout << "Average Load factor : " << totalLf << std::endl;
}



double rand_range(double min, double max) {
  return min + rand() * (max - min) / RAND_MAX;
}

uint32_t read_ADDR_from_file(std::string addrFile, std::map<uint32_t, std::map<uint32_t, addr_entry_t> > & ADDR) {
  std::ifstream fh_addrFile(addrFile.c_str());
  if (!fh_addrFile.is_open()) {
      std::cout << "read_ADDR_from_file() 无法打开文件" << addrFile <<std::endl;
      return 0;
  }
  std::vector<std::vector<std::string> > resLines;
  uint32_t lineCnt = read_files_by_line(fh_addrFile, resLines);
  NS_LOG_INFO("read_ADDR_from_file() reads file: " << addrFile << " about " << lineCnt << " Lines");

  uint32_t nodeIdx = UINT32_MAX, portIdx = UINT32_MAX, addrCnt=0;
  Ipv4Address network, mask, base;
  for (uint32_t i = 0; i < lineCnt; i++) {
    nodeIdx = string_to_integer(resLines[i][0]);
    portIdx = string_to_integer(resLines[i][1]);
    struct addr_entry_t *e = new addr_entry_t();
    e->network = string_to_ipv4Address(resLines[i][2]);
    Ipv4Mask mask(resLines[i][3].c_str());
    e->mask = mask;
    e->base = string_to_ipv4Address(resLines[i][4]);
    std::map<uint32_t, std::map<uint32_t, addr_entry_t> >::iterator it_0 = ADDR.find(nodeIdx);
    if (it_0 == ADDR.end()) {
        std::map<uint32_t, addr_entry_t> m;
        m[portIdx] = *e;
        ADDR[nodeIdx] = m;
        addrCnt = addrCnt + 1;
    }else{
      std::map<uint32_t, addr_entry_t>::iterator it_1 = it_0->second.find(portIdx);
      if (it_1 == it_0->second.end()) {
          it_0->second[portIdx] = *e;
          addrCnt = addrCnt + 1;
      }else{
          std::cout << "read_ADDR_from_file() reads file: " << addrFile << ", Error in repeated address ";
          std::cout << resLines[i][0] <<", "<< resLines[i][1] << ", " << resLines[i][2] <<", ";
          std::cout << resLines[i][3] <<", "<< resLines[i][4] << std::endl;
          return 0;
      }
    }
  }
  fh_addrFile.close();
  return addrCnt;
}

uint32_t read_CHL_from_file(std::string chlFile, std::map<uint32_t, CHL_entry_t> &CHL){
  std::ifstream fh_chlFile(chlFile.c_str());
  if (!fh_chlFile.is_open()) {
        std::cerr << "read_CHL_from_file() 无法打开文件 " << chlFile << std::endl;
        return 0;
  }
  std::vector<std::vector<std::string> > resLines;
  uint32_t lineCnt = read_files_by_line(fh_chlFile, resLines);
  uint32_t chlCnt = 0;
  for (uint32_t i = 0; i < lineCnt; i++) {
    uint32_t chlIdx = std::atoi(resLines[i][0].c_str());
    CHL_entry_t chlEntry;
    chlEntry.srcNodeIdx = std::atoi(resLines[i][1].c_str());
    chlEntry.dstNodeIdx = std::atoi(resLines[i][2].c_str());
    chlEntry.widthInGbps = std::atoi(resLines[i][3].c_str());
    chlEntry.delayInUs = std::atoi(resLines[i][4].c_str());
    chlEntry.queueType = resLines[i][5];
    chlEntry.queueSize = resLines[i][6];
    CHL[chlIdx] = chlEntry;
    chlCnt = chlCnt + 1;
  }
  fh_chlFile.close();
  return chlCnt;
}



uint32_t read_flows_from_file(std::string flowFile, std::map<uint32_t, flow_entry_t> & flows){
  std::ifstream fh_flowFile(flowFile.c_str());
  if (!fh_flowFile.is_open()) {
        std::cerr << "read_flows_from_file() 无法打开文件 " << flowFile << std::endl;
        return 0;
  }
  std::vector<std::vector<std::string> > resLines;
  uint32_t flowCnt = read_files_by_line(fh_flowFile, resLines);
  flows.clear();
  for (uint32_t i = 0; i < flowCnt; i++) {
    flow_entry_t flowEntry;
    flowEntry.idx = std::atoi(resLines[i][0].c_str());
    flowEntry.prioGroup = std::atoi(resLines[i][1].c_str());
    flowEntry.pktCnt = std::atoi(resLines[i][2].c_str());
    flowEntry.srcSvIdx = std::atoi(resLines[i][3].c_str());
    flowEntry.dstSvIdx = std::atoi(resLines[i][4].c_str());
    flowEntry.srcPort = std::atoi(resLines[i][5].c_str());
    flowEntry.dstPort = std::atoi(resLines[i][6].c_str());
    flows[flowEntry.idx] = flowEntry;
  }
  fh_flowFile.close();
  return flowCnt;
}


uint32_t read_files_by_line(std::ifstream& fh, std::vector<std::vector<std::string> > &resLines) {
    resLines.clear();
    uint32_t lineCnt = 0;
    std::string strLine;
    while (std::getline(fh, strLine)) { // 逐行读取文件内容
        std::istringstream iss(strLine);
        std::vector<std::string> resline;
        std::string str;
        while (iss >> str) { // 使用 istringstream 进行逐词解析
            resline.push_back(str); // 将每个单词加入到结果向量中
                    // std::cout << str << std::endl;

        }
        if (resline.size() > 0) {
          resLines.push_back(resline);
          lineCnt = lineCnt + 1;
        }
        

    }
    return lineCnt;
}

uint32_t read_PIT_from_file(std::string pitFile, std::map<uint32_t, std::map<uint32_t, PathData> >& PIT) {
  // std::srand(static_cast<unsigned int>(std::time(0)));  
  std::ifstream fh_pitFile(pitFile.c_str());
  if (!fh_pitFile.is_open()) {
        std::cerr << "read_PIT_from_file() 无法打开文件 " << pitFile << std::endl;
        return 0;
  }
  std::vector<std::vector<std::string> > resLines;
  uint32_t lineCnt = read_files_by_line(fh_pitFile, resLines);
  std::cout << "read_PIT_from_file() reads file: " << pitFile << " about " << lineCnt << " Lines" <<std::endl;
  uint32_t nodeIdx = UINT32_MAX, pid = UINT32_MAX, pitSize=0, priority = 0;
  std::vector<uint32_t> ports;
  for (uint32_t i = 0; i < lineCnt; i++) {
    nodeIdx = std::atoi(resLines[i][0].c_str());
    PathData path;
    pid = std::atoi(resLines[i][1].c_str());
    path.pid = pid;
    priority = std::atoi(resLines[i][2].c_str());
    path.priority = priority;
    uint32_t pathLength = (resLines[i].size()-3)/2;
    for (uint32_t j = 3; j < pathLength + 3; j++) {
      uint32_t portIdx = std::atoi(resLines[i][j].c_str());
      path.portSequence.push_back(portIdx);
    }
    for (uint32_t j = 3+pathLength; j < resLines[i].size(); j++) {
      uint32_t nodeId = std::atoi(resLines[i][j].c_str());
      path.nodeIdSequence.push_back(nodeId);
    }
    path.tsGeneration = NanoSeconds(0);
    path.tsProbeLastSend = NanoSeconds(0);
    path.tsLatencyLastSend = NanoSeconds(0);
    path.latency = LINK_LATENCY_IN_NANOSECOND * path.portSequence.size();
    PIT[nodeIdx][pid] = path;
    pitSize = pitSize + 1;
  }
  fh_pitFile.close();
  return pitSize;
}

uint32_t read_PST_from_file(std::string pstFile, std::map<uint32_t, std::map<PathSelTblKey, pstEntryData> >& PST) {
  // std::srand(static_cast<unsigned int>(std::time(0)));  
  std::ifstream fh_pstFile(pstFile.c_str());
  if (!fh_pstFile.is_open()) {
        std::cerr << "read_PST_from_file() 无法打开文件 " << pstFile << std::endl;
        return 0;
  }
  std::vector<std::vector<std::string> > resLines;
  uint32_t lineCnt = read_files_by_line(fh_pstFile, resLines);
  std::cout << "read_PST_from_file() reads file: " << pstFile << " about " << lineCnt << " Lines" <<std::endl;
  uint32_t nodeIdx = UINT32_MAX, pstSize=0;
  Ipv4Address srcTorAddr, dstTorAddr;
  std::vector<uint32_t> pids;
  for (uint32_t i = 0; i < lineCnt; i++) {
    nodeIdx = atoi(resLines[i][0].c_str());
    srcTorAddr = string_to_ipv4Address(resLines[i][1]);
    dstTorAddr = string_to_ipv4Address(resLines[i][2]);
    PathSelTblKey pstKey(srcTorAddr, dstTorAddr);
    struct pstEntryData *pstEntry = new pstEntryData();
    pstEntry->pathNum = resLines[i].size() - 4;    
    pstEntry->highestPriorityPathIdx = std::atoi(resLines[i][3].c_str());
    pids.clear();
    for (uint32_t j = 4; j < resLines[i].size(); j++) {
      uint32_t portIdx = std::atoi(resLines[i][j].c_str());
      pids.push_back(portIdx);
    }
    pstEntry->paths = pids;
    PST[nodeIdx][pstKey] = *pstEntry;
    pstSize = pstSize + 1;
  }
  fh_pstFile.close();
  return pstSize;
}



uint32_t read_TFC_from_file(std::string trafficFile, std::map<uint32_t, std::vector<tfc_entry_t> >& TFC) {
  // std::srand(static_cast<unsigned int>(std::time(0)));  
  std::ifstream fh_trafficFile(trafficFile.c_str());
  if (!fh_trafficFile.is_open()) {
        std::cerr << "read_TFC_from_file() 无法打开文件 " << trafficFile << std::endl;
        return 0;
  }
  std::vector<std::vector<std::string> > resLines;
  uint32_t lineCnt = read_files_by_line(fh_trafficFile, resLines);
  std::cout << "read_TFC_from_file() reads file: " << trafficFile << " about " << lineCnt << " Lines" <<std::endl;
  uint32_t srcNodeIdx = UINT32_MAX, trafficSize=0;

  for (uint32_t i = 0; i < lineCnt; i++) {
    tfc_entry_t tfcEntry;
    srcNodeIdx = std::atoi(resLines[i][0].c_str());
    tfcEntry.srcNodeIdx = srcNodeIdx;
    tfcEntry.dstNodeIdx = std::atoi(resLines[i][1].c_str());
    tfcEntry.loadfactor = std::atof(resLines[i][2].c_str());
    tfcEntry.capacityInGbps = std::atoi(resLines[i][3].c_str());
    tfcEntry.flowCount = 0;
    tfcEntry.bytesCount = 0;
    TFC[srcNodeIdx].push_back(tfcEntry);
    tfcEntry.print();
    trafficSize = trafficSize + 1;
  }
  fh_trafficFile.close();
  return trafficSize;
}

  uint32_t read_TOPO_from_file(std::string topoFile, TOPO_t &TOPO) {
  // std::srand(static_cast<unsigned int>(std::time(0)));  
  std::ifstream fh_topoFile(topoFile.c_str());
  if (!fh_topoFile.is_open()) {
        std::cerr << "read_TOPO_from_file() 无法打开文件 " << topoFile << std::endl;
        return 0;
  }
  std::vector<std::vector<std::string> > resLines;
  uint32_t lineCnt = read_files_by_line(fh_topoFile, resLines);
  // std::cout << "read_TOPO_from_file() reads file: " << topoFile << " about " << lineCnt << " Lines" <<std::endl;

  TOPO.swNum = 0;
  TOPO.svNum = 0;
  TOPO.allNum = 0;
  TOPO.swNodeIdx.clear();
  TOPO.svNodeIdx.clear();

  std::string nodeType;
  for (uint32_t i = 0; i < lineCnt; i++) {
    uint32_t nodeIdx = std::atoi(resLines[i][0].c_str());
    nodeType = resLines[i][1];
    if(nodeType == "sw"){
      TOPO.swNum = TOPO.swNum + 1;
      TOPO.swNodeIdx.push_back(nodeIdx);
    }else if (nodeType == "sv") {
      TOPO.svNum = TOPO.svNum + 1;
      TOPO.svNodeIdx.push_back(nodeIdx);
    }else{
      std::cout<< "Error in read_TOPO_from_file() about unknwon node type :" << nodeType <<std::endl;
    }
  }
  TOPO.allNum = TOPO.swNum + TOPO.svNum;
  fh_topoFile.close();
  return TOPO.allNum;
}

uint32_t read_VMT_from_file(std::string vmtFile, std::map<Ipv4Address, vmt_entry_t> & VMT) {
    // nodeIdx portIdx, addr
    std::ifstream fh_vmtFile(vmtFile.c_str());
    if (!fh_vmtFile.is_open()) {
        std::cout << "read_VMT_from_file() 无法打开文件" << vmtFile <<std::endl;
        return 0;
    }
    std::vector<std::vector<std::string> > resLines;
    uint32_t lineCnt = read_files_by_line(fh_vmtFile, resLines);
    // NS_LOG_INFO("read_VMT_from_file() reads file: " << vmtFile << " about " << lineCnt << " Lines");
    uint32_t vmtCnt=0;
    Ipv4Address svAddr, torAddr;
    for (uint32_t i = 0; i < lineCnt; i++) {
      svAddr = string_to_ipv4Address(resLines[i][0]);
      torAddr = string_to_ipv4Address(resLines[i][1]);
      VMT[svAddr].torAddr = torAddr;
      vmtCnt = vmtCnt + 1;
    }
  fh_vmtFile.close();
  return vmtCnt;
}

void screen_display(uint64_t displayIntervalInNs) {
  uint64_t curTimeInNs = Simulator::Now().GetNanoSeconds();
  std::cout << "The Current Time is " << curTimeInNs/1000 << "us\n" << std::flush;
  Simulator::Schedule(NanoSeconds(displayIntervalInNs), &screen_display, displayIntervalInNs);
}

PointToPointHelper set_P2P_attribute(uint32_t rate, uint32_t latency, std::string queueType, std::string queueSize) {
  PointToPointHelper p2p;
  std::string rateStr = to_string(rate) + "Gbps";
  std::string delayStr = to_string(latency) + "us";
  p2p.SetDeviceAttribute("DataRate", StringValue(rateStr));
  p2p.SetChannelAttribute("Delay", StringValue(delayStr));
  p2p.SetQueue(queueType, "MaxSize", QueueSizeValue(QueueSize(queueSize))); // 队列缓存大小
  // p2p.SetQueue(queueType,
  //             "Mode", StringValue(queueMode),
  //             queueUnit, UintegerValue(queueSize));
  return p2p;
}

uint32_t string_to_integer(const std::string& str) {
    std::istringstream iss(str);
    int number;
    if (!(iss >> number)) { // 检查是否成功转换
        // 如果字符串不能转换为有效的整数，则可以在这里处理错误情况，例如抛出异常或返回一个默认值
        std::cout<< "Invalid string for conversion to integer" << std::endl;
        return 0;
    }
    return number;
}

Ipv4Address string_to_ipv4Address(std::string str) {
  Ipv4Address ipAddress(str.c_str());
  return ipAddress;
}

uint32_t get_egress_port_from_pit_entry(PathData * pitEntry){
  if (pitEntry == 0)  {
    std::cout << "Error in get_egress_port_from_pit_entry() with null PitEntry" << std::endl;
    return 0;
  }else{
    if (pitEntry->portSequence.size() == 0) {
      std::cout << "Error in get_egress_port_from_pit_entry() with null portSequence" << std::endl;
      return 0;
    }
    return pitEntry->portSequence[0];
  }
  
}
std::vector<uint32_t> get_egress_ports_from_pit_entries(std::vector<PathData *> pitEntries){

  uint32_t n = pitEntries.size();
  std::vector<uint32_t> ports(n);
  for (size_t i = 0; i < n; i++) {
    ports[i] = get_egress_port_from_pit_entry(pitEntries[i]);
  }
  return ports;
}



// void print_mac_address_for_single_node(Ptr<Node> curNode){
//   Ptr<Ipv4> curIpv4 = curNode->GetObject<Ipv4> ();
//   uint32_t intfCnt = curIpv4->GetNInterfaces ();
//   Mac48AddressValue macAddressValue;
//   for (uint32_t j = 1; j < intfCnt; ++j) {
//     Ptr<NetDevice> netDevice = curIpv4->GetNetDevice (j);
//     netDevice->GetAttribute ("MacAddress", macAddressValue);
//     Mac48Address macAddress = macAddressValue.Get();
//     std::cout << "Node " << curNode->GetId () << ", Interface " << j << ", MAC Address: " << macAddress << std::endl;
//   }
//   return ;
// }

// msg="Attribute name=MacAddress does not exist for this object: tid=ns3::PointToPointNetDevice", +0.000000000s -1 file=../src/core/model/object-base.cc, line=230
// terminate called without an active exception

void monitor_switch_qlen(global_variable_t * varMap, uint32_t roundIdx){
    if (!varMap->enableQlenMonitor) {
      return ;
    }
    
    // NS_LOG_INFO("----------monitor_switch_qlen()----------");
    // NS_LOG_INFO("Queue Length Record File: "<< varMap->qlenMonitorFileName);
    if (varMap->qlenMonitorFileHandle == NULL) {
        varMap->qlenMonitorFileHandle = fopen(varMap->qlenMonitorFileName.c_str(), "w");
    }
    FILE * os = varMap->qlenMonitorFileHandle;
    if (os == NULL) {
      std::cout << "Error for Cannot open file " << varMap->qlenMonitorFileName << std::endl;
    }
    
    NodeContainer & swNodes = varMap->swNodes;
    uint64_t curTimeInNs = Simulator::Now().GetNanoSeconds();
    for (uint32_t i = 0; i < swNodes.GetN(); i++){
        if (swNodes.Get(i)->GetNodeType() == SWITCH_NODE_TYPE){ // is switch
            Ptr<SwitchNode> sw = DynamicCast<SwitchNode>(swNodes.Get(i));
            for (uint32_t j = 1; j < sw->GetNDevices(); j++){
                fprintf(os, "Round:%d TimeInNs:%lu Node:%d Nic:%d IngressLenInByte:[", roundIdx, curTimeInNs, i, j);
                uint32_t s = 0;
                for (uint32_t k = 0; k < SwitchMmu::qCnt; k++){
                    fprintf(os, "%d ", sw->m_mmu->ingress_bytes[j][k]);
                    s  = s + sw->m_mmu->ingress_bytes[j][k];
                }
                fprintf(os, "] LenSumInByte:%d egressLenInByte:[", s);
                s = 0;
                for (uint32_t k = 0; k < SwitchMmu::qCnt; k++){
                    fprintf(os, "%d ", sw->m_mmu->egress_bytes[j][k]);
                    s  = s + sw->m_mmu->egress_bytes[j][k];
                }
                fprintf(os, "] LenSumInByte:%d\n", s);

            }
        }
    }
    fflush(os);

    if ((curTimeInNs + varMap->qlenMonitorIntervalInNs) < (varMap->simEndTimeInSec * 1000000000) &&(!Simulator::IsFinished()) ){
        Simulator::Schedule(NanoSeconds(varMap->qlenMonitorIntervalInNs), &monitor_switch_qlen, varMap, roundIdx+1);
    }
}


void monitor_special_port_qlen(global_variable_t * varMap, uint32_t nodeId, uint32_t portId, uint32_t roundIdx){
    if (!varMap->enableQlenMonitor) {
      return ;
    }
    
    // NS_LOG_INFO("----------monitor_switch_qlen()----------");
    // NS_LOG_INFO("Queue Length Record File: "<< varMap->qlenMonitorFileName);
    if (varMap->qlenMonitorFileHandle == NULL) {
        varMap->qlenMonitorFileHandle = fopen(varMap->qlenMonitorFileName.c_str(), "w");
    }
    FILE * os = varMap->qlenMonitorFileHandle;
    if (os == NULL) {
      std::cout << "Error for Cannot open file " << varMap->qlenMonitorFileName << std::endl;
    }
    
    NodeContainer & swNodes = varMap->swNodes;
    uint64_t curTimeInNs = Simulator::Now().GetNanoSeconds();
    for (uint32_t i = 0; i < swNodes.GetN(); i++){
        if (i != nodeId) {
          continue;
        }
        
        if (swNodes.Get(i)->GetNodeType() == SWITCH_NODE_TYPE){ // is switch
            Ptr<SwitchNode> sw = DynamicCast<SwitchNode>(swNodes.Get(i));
            for (uint32_t j = 1; j < sw->GetNDevices(); j++){
              if (j != portId) {
                continue;
              }
              
                fprintf(os, "Round:%d TimeInNs:%lu Node:%d Nic:%d IngressLenInByte:[", roundIdx, curTimeInNs, i, j);
                uint32_t s = 0;
                for (uint32_t k = 0; k < SwitchMmu::qCnt; k++){
                    fprintf(os, "%d ", sw->m_mmu->ingress_bytes[j][k]);
                    s  = s + sw->m_mmu->ingress_bytes[j][k];
                }
                fprintf(os, "] LenSumInByte:%d egressLenInByte:[", s);
                s = 0;
                for (uint32_t k = 0; k < SwitchMmu::qCnt; k++){
                    fprintf(os, "%d ", sw->m_mmu->egress_bytes[j][k]);
                    s  = s + sw->m_mmu->egress_bytes[j][k];
                }
                fprintf(os, "] LenSumInByte:%d\n", s);

            }
        }
    }
    fflush(os);

    if ((varMap->numOfFinishedJob < varMap->kvCachePara.size())&&(curTimeInNs + varMap->qlenMonitorIntervalInNs) < (varMap->simEndTimeInSec * 1000000000) &&(!Simulator::IsFinished()) ){
        Simulator::Schedule(NanoSeconds(varMap->qlenMonitorIntervalInNs), &monitor_special_port_qlen, varMap, nodeId, portId, roundIdx+1);
    }
}



void monitor_pfc(FILE * os, Ptr<QbbNetDevice> dev, uint32_t type){
    uint64_t curTimeInNs = Simulator::Now().GetNanoSeconds();
    uint32_t nodeId = dev->GetNode()->GetId();
    uint32_t nodeType = dev->GetNode()->GetNodeType();
    uint32_t nicIdx = dev->GetIfIndex();
    // fprintf(os, "%lu %d %d %d", curTimeInNs, nodeId, nodeType, nicIdx);
    if (FLAG_PFC_PAUSE == type) {
      fprintf(os, "TimeInNs:%lu Node:%d NodeType:%d Nic:%d PfcType:%s\n", curTimeInNs, nodeId, nodeType, nicIdx, "PAUSE");
    }else if (FLAG_PFC_RESUME == type) {
      fprintf(os, "TimeInNs:%lu Node:%d NodeType:%d Nic:%d PfcType:%s\n", curTimeInNs, nodeId, nodeType, nicIdx,"RESUME");
    }else {
      fprintf(os, "TimeInNs:%lu Node:%d NodeType:%d Nic:%d PfcType:%s\n", curTimeInNs, nodeId, nodeType, nicIdx,"UNKOWN");
    }
    fflush(os);
    return ;
}

NodeContainer merge_nodes(NodeContainer & first, NodeContainer & second){
  NodeContainer all;
  all.Add(first);
  all.Add(second);
  return all;
}

CHL_entry_t parse_channel_entry(std::vector<std::string> & s){
    CHL_entry_t chl;
    chl.chlIdx = std::atoi(s[0].c_str());
    chl.srcNodeIdx = std::atoi(s[1].c_str());
    chl.dstNodeIdx = std::atoi(s[2].c_str());
    chl.widthInGbps = std::atoi(s[3].c_str());
    chl.delayInUs = std::atoi(s[4].c_str());
    return chl;
}

std::map<uint32_t, CHL_entry_t> parse_channels( std::vector<std::vector<std::string> > & resLines){
  std::map<uint32_t, CHL_entry_t> m;
  for (uint32_t i = 0; i < resLines.size(); i++) {
    CHL_entry_t chl = parse_channel_entry(resLines[i]);
    m[chl.chlIdx] = chl;
  }
  return m;
}

void create_topology_rdma(global_variable_t * varMap) {
  std::ifstream fh(varMap->topoFileName.c_str());
  if (!fh.is_open()) {
      std::cout << "create_topology_rdma() 无法打开文件" << varMap->topoFileName <<std::endl;
      return ;
  }
  std::vector<std::vector<std::string> > resLines;
  read_files_by_line(fh, resLines);
  fh.close();

  varMap->swNodes = create_nodes<SwitchNode>(std::atoi(resLines[0][0].c_str()));
  varMap->svNodes = create_nodes<Node>(std::atoi(resLines[0][1].c_str()));
  varMap->allNodes = merge_nodes(varMap->swNodes, varMap->svNodes);
  std::vector<std::vector<std::string> > s(resLines.begin()+1, resLines.end());
  varMap->channels = parse_channels(s);
  add_QBB_channels(varMap);
  install_rdma_driver(varMap);
  return ;
}




QbbHelper set_QBB_attribute(uint32_t rate, uint32_t latency) {
  QbbHelper qbb;
  std::string rateStr = to_string(rate) + "Gbps";
  std::string delayStr = to_string(latency) + "us";
  qbb.SetDeviceAttribute("DataRate", StringValue(rateStr));
  qbb.SetChannelAttribute("Delay", StringValue(delayStr));
  return qbb;
}


void add_QBB_channels(global_variable_t * varMap){
  InternetStackHelper internet;
  internet.Install(varMap->allNodes);
  std::map<Ptr<Node>, std::map<Ptr<Node>, std::vector<edge_t> > > & edges = varMap->edges;
  NodeContainer &allNodes = varMap->allNodes;
  std::map<uint32_t, CHL_entry_t> &CHL = varMap->channels;
  FILE * os;
  if (varMap->enablePfcMonitor) {
    varMap->pfcFileHandle = fopen(varMap->pfcFileName.c_str(), "w");
    os = varMap->pfcFileHandle;
    if (!os) {
      std::cout << "add_QBB_channels() cannot open file " << varMap->pfcFileName << std::endl;
      return ;
    }
  }
  uint32_t channelCnt = CHL.size();
  for (uint32_t i = 0; i < channelCnt; i++) {
    CHL_entry_t channelEntry = CHL[i];
    channelEntry.print();
    uint32_t srcNodeIdx = channelEntry.srcNodeIdx;
    Ptr<Node> srcNode = allNodes.Get(srcNodeIdx);
    uint32_t dstNodeIdx = channelEntry.dstNodeIdx;
    Ptr<Node> dstNode = allNodes.Get(dstNodeIdx);
    QbbHelper qbb = set_QBB_attribute(channelEntry.widthInGbps, channelEntry.delayInUs);
    NetDeviceContainer d = qbb.Install(srcNode, dstNode);
    if (varMap->enablePfcMonitor) {
        DynamicCast<QbbNetDevice>(d.Get(0))->TraceConnectWithoutContext("QbbPfc", MakeBoundCallback (&monitor_pfc, os, DynamicCast<QbbNetDevice>(d.Get(0))));
        DynamicCast<QbbNetDevice>(d.Get(1))->TraceConnectWithoutContext("QbbPfc", MakeBoundCallback (&monitor_pfc, os, DynamicCast<QbbNetDevice>(d.Get(1))));
    }

    // used to create a graph of the topology
    edge_t e1;
    e1.nicIdx = DynamicCast<QbbNetDevice>(d.Get(0))->GetIfIndex();
    e1.up = true;
    e1.delayInNs = DynamicCast<QbbChannel>(DynamicCast<QbbNetDevice>(d.Get(0))->GetChannel())->GetDelay().GetNanoSeconds();
    e1.bwInBitps = DynamicCast<QbbNetDevice>(d.Get(0))->GetDataRate().GetBitRate();
    edges[srcNode][dstNode].push_back(e1);

    // edges[srcNode][dstNode].nicIdx = DynamicCast<QbbNetDevice>(d.Get(0))->GetIfIndex();
    // edges[srcNode][dstNode].up = true;
    // edges[srcNode][dstNode].delayInNs = DynamicCast<QbbChannel>(DynamicCast<QbbNetDevice>(d.Get(0))->GetChannel())->GetDelay().GetNanoSeconds();
    // edges[srcNode][dstNode].bwInBitps = DynamicCast<QbbNetDevice>(d.Get(0))->GetDataRate().GetBitRate();
    // edges[dstNode][srcNode].nicIdx = DynamicCast<QbbNetDevice>(d.Get(1))->GetIfIndex();
    // edges[dstNode][srcNode].up = true;
    // edges[dstNode][srcNode].delayInNs = DynamicCast<QbbChannel>(DynamicCast<QbbNetDevice>(d.Get(1))->GetChannel())->GetDelay().GetNanoSeconds();
    // edges[dstNode][srcNode].bwInBitps = DynamicCast<QbbNetDevice>(d.Get(1))->GetDataRate().GetBitRate();
    edge_t e2;
    e2.nicIdx = DynamicCast<QbbNetDevice>(d.Get(1))->GetIfIndex();
    e2.up = true;
    e2.delayInNs = DynamicCast<QbbChannel>(DynamicCast<QbbNetDevice>(d.Get(1))->GetChannel())->GetDelay().GetNanoSeconds();
    e2.bwInBitps = DynamicCast<QbbNetDevice>(d.Get(1))->GetDataRate().GetBitRate();
    edges[dstNode][srcNode].push_back(e2);


  }
  return ;
}



uint64_t get_nic_rate_In_Gbps(global_variable_t * varMap){
    Ptr<Node> node = varMap->svNodes.Get(0);
    return DynamicCast<QbbNetDevice>(node->GetDevice(1))->GetDataRate().GetBitRate()/1000000000;

}

void config_switch_mmu(global_variable_t * varMap){
  NodeContainer & swNodes =  varMap->swNodes;
  std::map<uint32_t, ecn_para_entry_t>& ecnParaMap = varMap->ecnParas;
  uint64_t nicRateInGbps = get_nic_rate_In_Gbps(varMap);
    uint32_t node_num = swNodes.GetN();
    for (uint32_t nodeIdx = 0; nodeIdx < node_num; nodeIdx++){
        Ptr<SwitchNode> swNode = DynamicCast<SwitchNode>(swNodes.Get(nodeIdx));
        uint32_t swNodeId = swNode->GetId();
        if (swNode->GetNodeType() != SWITCH_NODE_TYPE){
            std::cout << "Error in config_mmu_switch with nodeId: " << swNodeId << std::endl;
            return ;
        }

        for (uint32_t nicIdx = 1; nicIdx < swNode->GetNDevices(); nicIdx++){
            Ptr<QbbNetDevice> dev = DynamicCast<QbbNetDevice>(swNode->GetDevice(nicIdx));
            // set ecn
            uint64_t rateInBit = dev->GetDataRate().GetBitRate();
            uint32_t rateInGbps = rateInBit / 1000000000;

            swNode->m_mmu->ConfigEcn(nicIdx, varMap->ecnParaMap[rateInGbps].kminInKb, ecnParaMap[rateInGbps].kmaxInKb, ecnParaMap[rateInGbps].pmax);
            // set pfc
            uint64_t delayInNs = DynamicCast<QbbChannel>(dev->GetChannel())->GetDelay().GetNanoSeconds();
            uint32_t headroomInByte = rateInBit / 1000000000 * delayInNs / 8 * 3; // 8是指byte
            swNode->m_mmu->ConfigHdrm(nicIdx, headroomInByte);

            // set pfc alpha, proportional to link bw, larger bw indicates large utilization
            swNode->m_mmu->pfc_a_shift[nicIdx] = varMap->alphaShiftInLog;
            while (rateInGbps > nicRateInGbps && swNode->m_mmu->pfc_a_shift[nicIdx] > 0){
                swNode->m_mmu->pfc_a_shift[nicIdx]--;
                rateInGbps /= 2;
            }
        }
        swNode->m_mmu->ConfigNPort(swNode->GetNDevices()-1);
        swNode->m_mmu->ConfigBufferSize(varMap->mmuSwBufferSizeInMB* 1024 * 1024);
        swNode->m_mmu->node_id = swNode->GetId();

        
    }

}

void config_switch(global_variable_t * varMap){
    config_switch_mmu(varMap);
    set_switch_cc_para(varMap);
    return ;
}


std::vector<std::string> stringSplitWithTargetChar(const std::string &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}




void install_rdma_driver(global_variable_t * varMap){
  varMap->fctFileHandle = fopen(varMap->fctFileName.c_str(), "w");
  if (!varMap->fctFileHandle) {
    std::cout << "config_rdma_nic() cannot open file " << varMap->fctFileName << std::endl;
    return ;
  }
  NodeContainer & svNodes = varMap->svNodes;
  nic_para_entry_t& nicParas = varMap->nicParas;
  uint32_t svNum = svNodes.GetN();
    for (uint32_t svIdx = 0; svIdx < svNum; svIdx++){
      Ptr<Node> svNode = svNodes.Get(svIdx);
        if (svNode->GetNodeType() == SERVER_NODE_TYPE){ // is server
            // create RdmaHw
            Ptr<RdmaHw> rdmaHw = CreateObject<RdmaHw>();
            rdmaHw->SetAttribute("ClampTargetRate", BooleanValue(nicParas.enableClampTargetRate));
            rdmaHw->SetAttribute("AlphaResumInterval", DoubleValue(nicParas.alphaResumeIntervalInUs));
            rdmaHw->SetAttribute("RPTimer", DoubleValue(nicParas.rpTimerInUs));
            rdmaHw->SetAttribute("FastRecoveryTimes", UintegerValue(nicParas.fastRecoveryTimes));
            rdmaHw->SetAttribute("EwmaGain", DoubleValue(nicParas.ewmaGain));
            rdmaHw->SetAttribute("RateAI", DataRateValue(DataRate(to_data_rate(nicParas.rateAiInMbps, "Mb/s"))));
            rdmaHw->SetAttribute("RateHAI", DataRateValue(DataRate(to_data_rate(nicParas.rateHaiInMbps, "Mb/s"))));
            rdmaHw->SetAttribute("L2BackToZero", BooleanValue(nicParas.enableL2BackToZero));
            rdmaHw->SetAttribute("L2ChunkSize", UintegerValue(nicParas.l2ChunkSizeInByte));
            rdmaHw->SetAttribute("L2AckInterval", UintegerValue(nicParas.l2AckIntervalInByte));
            rdmaHw->SetAttribute("CcMode", UintegerValue(varMap->ccMode));
            rdmaHw->SetAttribute("RateDecreaseInterval", DoubleValue(nicParas.rateDecreaseIntervalInUs));
            rdmaHw->SetAttribute("MinRate", DataRateValue(DataRate(to_data_rate(nicParas.minRateInMbps, "Mb/s"))));
            rdmaHw->SetAttribute("Mtu", UintegerValue(varMap->pktPayloadSizeInByte));
            rdmaHw->SetAttribute("MiThresh", UintegerValue(nicParas.miThresh));
            rdmaHw->SetAttribute("VarWin", BooleanValue(nicParas.enableVarWin));
            rdmaHw->SetAttribute("FastReact", BooleanValue(nicParas.enableFastFeedback));
            rdmaHw->SetAttribute("MultiRate", BooleanValue(nicParas.enableMultiRate));
            rdmaHw->SetAttribute("SampleFeedback", BooleanValue(nicParas.enableSampleFeedback));
            rdmaHw->SetAttribute("TargetUtil", DoubleValue(nicParas.targetUtilization));
            rdmaHw->SetAttribute("RateBound", BooleanValue(nicParas.enableRateBound));
            rdmaHw->SetAttribute("DctcpRateAI", DataRateValue(DataRate(to_data_rate(nicParas.dctcpRateAiInMbps, "Mb/s"))));
            rdmaHw->SetPintSmplThresh(nicParas.pintProbTresh);
            // create and install RdmaDriver
            Ptr<RdmaDriver> rdma = CreateObject<RdmaDriver>();
            rdma->SetNode(svNode);
            rdma->SetRdmaHw(rdmaHw);

            svNode->AggregateObject (rdma);
            rdma->Init();
            if (varMap->enableKvCache) {
              rdma->TraceConnectWithoutContext("QpComplete", MakeBoundCallback (&qp_finish_kv_cache, varMap->fctFileHandle, varMap));
            }else{
              rdma->TraceConnectWithoutContext("QpComplete", MakeBoundCallback (&qp_finish, varMap->fctFileHandle, varMap));
            }
            
        }
    }

    // set ACK priority on hosts
    if (varMap->enableAckHigherPrio){
        RdmaEgressQueue::ack_q_idx = 0;
    }else{
        RdmaEgressQueue::ack_q_idx = 3;
    }
}


 
bool stringToBool(const std::string& str) {
    std::string lowerStr = str;
    for (char &c : lowerStr) {
        c = std::tolower(c); // 转换为小写
    }
 
    if (lowerStr == "true" || lowerStr == "1" || lowerStr == "yes" || lowerStr == "on") {
        return true;
    } else if (lowerStr == "false" || lowerStr == "0" || lowerStr == "no" || lowerStr == "off") {
        return false;
    }
    return false;
}

  void calculate_paths_to_single_server(global_variable_t * varMap, Ptr<Node> host){

    std::map<Ptr<Node>, std::map<Ptr<Node>, std::vector<edge_t> > > &edges = varMap->edges;
    // queue for the BFS.
    std::vector<Ptr<Node> > q;
    // Distance from the host to each node.
    std::map<Ptr<Node>, int> dis;
    std::map<Ptr<Node>, uint64_t> delayInNs;
    std::map<Ptr<Node>, uint64_t> txDelayInNs;
    std::map<Ptr<Node>, uint64_t> bwInBitps;
    // init BFS.
    q.push_back(host);
    dis[host] = 0;
    delayInNs[host] = 0;
    txDelayInNs[host] = 0;
    bwInBitps[host] = 0xfffffffffffffffflu;
    // BFS.
    for (int i = 0; i < (int)q.size(); i++){
        Ptr<Node> now = q[i];
        int d = dis[now];
        for (auto it = edges[now].begin(); it != edges[now].end(); it++){
            // skip down link
            // if (!it->second.up)
            //     continue;
            Ptr<Node> next = it->first;
            // If 'next' have not been visited.
            if (dis.find(next) == dis.end()){
                dis[next] = d + 1;
                uint64_t maxDelayInNs = 0, minBwInBitps = 0xfffffffffffffffflu;
                for (uint32_t i = 0; i < it->second.size(); i++) {
                  maxDelayInNs = std::max(maxDelayInNs, it->second[i].delayInNs);
                  minBwInBitps = std::min(minBwInBitps, it->second[i].bwInBitps);
                }
                
                delayInNs[next] = delayInNs[now] + maxDelayInNs;
                txDelayInNs[next] = txDelayInNs[now] + varMap->defaultPktSizeInByte * 1000000000lu * 8 / minBwInBitps;
                bwInBitps[next] = std::min(bwInBitps[now], minBwInBitps);
                // we only enqueue switch, because we do not want packets to go through host as middle point
                if (next->GetNodeType() == SWITCH_NODE_TYPE)
                    q.push_back(next);
            }
            // if 'now' is on the shortest path from 'next' to 'host'.
            if (d + 1 == dis[next]){
                varMap->nextHop[next][host].push_back(now);
            }
        }
    }
    for (auto it : delayInNs){
        varMap->pairDelayInNs[it.first][host] = it.second;
    }
    for (auto it : txDelayInNs){
        varMap->pairTxDelayInNs[it.first][host] = it.second;
    }
    for (auto it : bwInBitps){
        varMap->pairBwInBitps[it.first][host] = it.second;
    }
    return ;
}

  uint32_t calculate_paths_for_servers(global_variable_t * varMap){
    NodeContainer svNodes = varMap->svNodes;
    uint32_t hostCnt = 0;
    for (uint32_t i = 0; i < svNodes.GetN(); i++) {
        Ptr<Node> svNode = svNodes.Get(i);
        if (svNode->GetNodeType() == SERVER_NODE_TYPE){
            calculate_paths_to_single_server(varMap, svNode);
            hostCnt += 1;    
        }else{
          std::cout << "Error in calculate_paths_for_servers() with wrong node type" << std::endl;
          return 0;
        }
    }
    return hostCnt;
  }

void install_routing_entries(global_variable_t * varMap){
    NS_LOG_INFO("----------Install The Routing Entries----------");
    print_connections_for_nodes(varMap->allNodes);
    std::map<Ptr<Node>, std::map<Ptr<Node>, std::vector<Ptr<Node> > > > & nextHop = varMap->nextHop;
    // For each node.
    for (auto i = nextHop.begin(); i != nextHop.end(); i++){
        Ptr<Node> node = i->first;
        auto &table = i->second;
        for (auto j = table.begin(); j != table.end(); j++){
            NS_LOG_INFO("CurNode: " << i->first->GetId() << ", DstServerNode: " << j->first->GetId());
            // The destination node.
            Ptr<Node> dst = j->first;
            // The IP address of the dst.
            Ipv4Address dstAddr = dst->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();
            // The next hops towards the dst.
            std::vector<Ptr<Node> > & nexts = j->second;
            std::vector<uint32_t> ports;
            for (int k = 0; k < (int)nexts.size(); k++){
                Ptr<Node> next = nexts[k];
                for (uint32_t p = 0; p < varMap->edges[node][next].size(); p++) {
                  uint32_t interface = varMap->edges[node][next][p].nicIdx;
                  ports.push_back(interface);
                  if (node->GetNodeType() == SWITCH_NODE_TYPE){
                      DynamicCast<SwitchNode>(node)->AddTableEntry(dstAddr, interface);
                  }
                  else{
                      node->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(dstAddr, interface);
                  }
                }
            }
        NS_LOG_INFO("CurNode: " << i->first->GetId() << ", DstServerNode: " << j->first->GetId() << ", Ports: " << vector_to_string(ports));
        // std:: cout << "CurNode: " << i->first->GetId() << ", DstServerNode: " << j->first->GetId() << ", Ports: " << vector_to_string(ports) <<std::endl;

        }
    }
}




void calculate_bdp_and_rtt(global_variable_t * varMap){
    varMap->maxRttInNs = varMap->maxBdpInByte = 0;
    auto & svNodes = varMap->svNodes;
    uint32_t svNodeCnt = svNodes.GetN();
    for (uint32_t i = 0; i < svNodeCnt; i++){
        for (uint32_t j = 0; j < svNodeCnt; j++) {
            if (i == j) {
              continue;
            }
            auto srcSvNode = svNodes.Get(i);
            auto dstSvNode = svNodes.Get(j);
            if ((srcSvNode->GetNodeType() != SERVER_NODE_TYPE) || (dstSvNode->GetNodeType() != SERVER_NODE_TYPE) ){
                std::cout << "Error in calculate_bdp_and_rtt() with wrong node type" << std::endl;
                return ;
            }

            uint64_t delayInNs = varMap->pairDelayInNs[srcSvNode][dstSvNode];
            uint64_t txDelayInNs = varMap->pairTxDelayInNs[srcSvNode][dstSvNode];
            uint64_t rttInNs = delayInNs * 2 + txDelayInNs;
            uint64_t bwInBitps = varMap->pairBwInBitps[srcSvNode][dstSvNode];
            uint64_t bdpInByte = rttInNs * bwInBitps / 1000000000/8; 
            varMap->pairBdpInByte[srcSvNode][dstSvNode] = bdpInByte;
            varMap->pairRttInNs[srcSvNode][dstSvNode] = rttInNs;
            if (bdpInByte > varMap->maxBdpInByte){
                varMap->maxBdpInByte = bdpInByte;
            }
            if (rttInNs > varMap->maxRttInNs){
                varMap->maxRttInNs = rttInNs;
            }
        }
    }
}

void set_switch_cc_para(global_variable_t * varMap){
    calculate_bdp_and_rtt(varMap);
    auto swNodes = varMap->swNodes;
    uint32_t swNodeCnt = swNodes.GetN();
    for (uint32_t i = 0; i < swNodeCnt; i++) {
        auto swNode = swNodes.Get(i);
        if (swNode->GetNodeType() == SWITCH_NODE_TYPE) { // switch
            Ptr<SwitchNode> sw = DynamicCast<SwitchNode>(swNode);
            sw->SetAttribute("CcMode", UintegerValue(varMap->ccMode));
            sw->SetAttribute("MaxRtt", UintegerValue(varMap->maxRttInNs));
            swNode->SetAttribute("EcnEnabled", BooleanValue(varMap->enableQcn));

        }else{
            std::cout << "Error in set_switch_cc_mode() with wrong node type" << std::endl;
            return ;
        }
    }
}

void set_QBB_trace(global_variable_t * varMap){
  if (varMap->enableQbbTrace) {
          std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4"<< std::endl;

    for (auto & e : varMap->traceNodeIds) {
      std::cout << "trace Node Id " << e << std::endl;
      varMap->traceNodes.Add(varMap->allNodes.Get(e));
    }
    
      NodeContainer & traceNodes = varMap->traceNodes;

      varMap->qbbFileHandle = fopen(varMap->qbbFileName.c_str(), "w");
      QbbHelper qbb;
      qbb.EnableTracing(varMap->qbbFileHandle, traceNodes);  
    }
  

    return ;
}

// void print_nic_info(global_variable_t * varMap){

//     varMap->nicInfoOutputFileHandle.open(varMap->nicInfoOutputFileName.c_str());
//     if (!varMap->nicInfoOutputFileHandle.is_open()) {
//       std::cout << "print_nic_info() cannot open file " << varMap->nicInfoOutputFileName << std::endl;
//       return ;
//     }

//     uint32_t allNodeCnt = varMap->allNodes.GetN();
//     for (uint32_t i = 0; i < allNodeCnt; i++) {
//       os << "Node: " << i <<", ";
//       auto curNode = varMap->allNodes.Get(i);
//       uint32_t nicCnt = curNode->GetNDevices();
//       for (uint32_t j = 1; j < nicCnt; j++) {
//           varMap->nicInfoOutputFileHandle << "NIC: " << j <<", ";
//           uint64_t rateInMBps = DynamicCast<QbbNetDevice>(curNode->GetDevice(j))->GetDataRate().GetBitRate() / 8000000lu ;
//           varMap->nicInfoOutputFileHandle << "MBps: " << rateInMBps << std::endl;
//       } 
//     }
// }


void generate_rdma_flows_for_node_pair(global_variable_t * varMap) {
  // std::cout << "startTimeInSec:" << varMap->simStartTimeInSec << ", flowLunchEndTimeInSec: " <<varMap->flowLunchEndTimeInSec<< std::endl;
  double startTimeInSec = varMap->simStartTimeInSec + poission_gen_interval(varMap->requestRate); // possion distribution of start time
  // std::cout << "NewFlowStartTimeInSec:" << startTimeInSec << std::endl;
  while (startTimeInSec < varMap->flowLunchEndTimeInSec) {
    // std::cout << "NewFlowStartTimeInSec:" << startTimeInSec << std::endl;
    flow_entry_t genFlow;
    // std::cout << "startTime :" << startTime << ", FLOW_LAUNCH_END_TIME : " << FLOW_LAUNCH_END_TIME << ", END_TIME : " << END_TIME;
    genFlow.srcNode = varMap->srcNode;
    genFlow.dstNode = varMap->dstNode;
    genFlow.srcPort = varMap->appStartPort + 1;
    genFlow.dstPort = varMap->appStartPort + 1;
    varMap->appStartPort = varMap->appStartPort + 1;
    genFlow.startTimeInSec  = startTimeInSec;
    genFlow.srcAddr = genFlow.srcNode->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();
    genFlow.dstAddr = genFlow.dstNode->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();
    genFlow.prioGroup = varMap->flowGroupPrio;
    genFlow.byteCnt = gen_random_cdf(varMap->cdfTable);

    if (!varMap->enableWindow) {
      genFlow.winInByte = 0;
    }else{
      if (varMap->enableUnifiedWindow) {
        genFlow.winInByte = varMap->maxBdpInByte;
        /* code */
      }else{
        genFlow.winInByte = varMap->pairBdpInByte[genFlow.srcNode][genFlow.dstNode];
      }
    }

    if (varMap->enableUnifiedWindow) {
      genFlow.rttInNs = varMap->maxRttInNs;
      /* code */
    }else{
      genFlow.rttInNs = varMap->pairRttInNs[genFlow.srcNode][genFlow.dstNode];
    }
    varMap->flowCount = varMap->flowCount + 1;
    genFlow.idx = varMap->flowCount;
    varMap->genFlows.push_back(genFlow);
    varMap->totalFlowSizeInByte = varMap->totalFlowSizeInByte + genFlow.byteCnt;
    if (genFlow.byteCnt <= varMap->smallFlowThreshInByte) {
      varMap->smallFlowCount = varMap->smallFlowCount + 1;
    }else if (genFlow.byteCnt >= varMap->largeFLowThreshInByte) {
      varMap->largeFlowCount = varMap->largeFlowCount + 1;
    }
    startTimeInSec = startTimeInSec + poission_gen_interval(varMap->requestRate);
  }
}

flow_entry_t generate_single_rdma_flow(uint32_t flowId, Ptr<Node> srcNode, Ptr<Node> dstNode,
                              uint32_t port, uint32_t flowSize, double START_TIME, uint32_t flowPg, uint64_t winInByte, uint64_t rttInNs) {
    flow_entry_t genFlow;
    genFlow.srcNode = srcNode;
    genFlow.dstNode = dstNode;
    genFlow.srcPort = port;
    genFlow.dstPort = port;
    genFlow.startTimeInSec  = START_TIME;
    genFlow.srcAddr = srcNode->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();
    genFlow.dstAddr = dstNode->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();
    genFlow.prioGroup = flowPg;
    genFlow.byteCnt = flowSize;
    genFlow.winInByte = winInByte;
    genFlow.rttInNs = rttInNs;
    genFlow.idx = flowId;
    return genFlow;  
}


void generate_rdma_flows_on_nodes(global_variable_t * varMap) {
  NS_LOG_INFO("----------Generate RDMA Flows On Nodes----------");
  // source.SetAttribute ("SendSize", UintegerValue (PACKET_SIZE));
  auto it_0 = varMap->tfc.begin();
  for (; it_0 != varMap->tfc.end(); it_0++) {
    varMap->srcNode = varMap->allNodes.Get(it_0->first);
    std::vector<tfc_entry_t> tfcEntries = it_0->second;
    // std::cout << "srcNode ID: " <<  srcNode->GetId() << construct_target_string(5, " ");
    for (uint32_t i = 0; i < tfcEntries.size(); i++) {
      // std::cout << "sequence : " << i << construct_target_string(5, " ");
      tfc_entry_t tfcEntry = tfcEntries[i];
      uint32_t dstNodeIdx = tfcEntry.dstNodeIdx;
      varMap->dstNode = varMap->allNodes.Get(dstNodeIdx);
      // std::cout << "dstNode ID: " <<  dstNode->GetId() << construct_target_string(5, " ");
      // std::cout << "loadfactorAdjustFacror: " <<  loadfactorAdjustFacror << construct_target_string(5, " ");
      double tmpLoadFactor = varMap->loadRatioShift * varMap->loadRatio;
      // std::cout << "tmpLoadFactor: " <<  tmpLoadFactor << construct_target_string(5, " ");
      uint64_t rateInBitPerSec = DynamicCast<QbbNetDevice>(varMap->srcNode->GetDevice(1))->GetDataRate().GetBitRate() ;
      // std::cout << "widthInGbps: " <<  widthInGbps << construct_target_string(5, " ");
      varMap->requestRate = tmpLoadFactor * rateInBitPerSec / 8 / avg_cdf (varMap->cdfTable);
      NS_LOG_INFO("srcNode: " << it_0->first << ", " <<
                  "DstNode: " << dstNodeIdx << ", " <<
                  "loadRatio: " << varMap->loadRatio << ", " <<
                  "loadRatioShift: " << varMap->loadRatioShift << ", " <<
                  "rateInBitPerSec: " << rateInBitPerSec << ", " <<
                  "RequestRate: " <<  varMap->requestRate);
      generate_rdma_flows_for_node_pair(varMap);
    }
  }
}

void install_rdma_flows_on_nodes(global_variable_t * varMap) {
  auto & flows = varMap->genFlows;
  for (uint32_t i = 0; i < flows.size(); i++) {
    auto & flow = flows[i];
    flow.print();
    RdmaClientHelper clientHelper(flow.prioGroup, flow.srcAddr, flow.dstAddr, flow.srcPort, flow.dstPort, flow.byteCnt, flow.winInByte, flow.rttInNs);
    ApplicationContainer appCon = clientHelper.Install(flow.srcNode);
    appCon.Start(Seconds(flow.startTimeInSec));

  }
}

void install_rdma_application(global_variable_t * varMap) {
    load_cdf(varMap->cdfTable, varMap->workLoadFileName.c_str());
    read_TFC_from_file(varMap->tfcFileName, varMap->tfc);
    generate_rdma_flows_on_nodes(varMap);
    install_rdma_flows_on_nodes(varMap);
}

void install_rdma_test_application(global_variable_t * varMap, uint32_t srcNodeId, uint32_t dstNodeId, uint32_t port, uint32_t flowSize, double startTime ) {

    auto srcNode =  varMap->allNodes.Get(srcNodeId);
    auto dstNode = varMap->allNodes.Get(dstNodeId);
    uint64_t winInByte, rttInNs;
    if (!varMap->enableWindow) {
      winInByte = 0;
    }else{
      if (varMap->enableUnifiedWindow) {
        winInByte = varMap->maxBdpInByte;
        /* code */
      }else{
        winInByte = varMap->pairBdpInByte[srcNode][dstNode];
      }
    }

    if (varMap->enableUnifiedWindow) {
      rttInNs = varMap->maxRttInNs;
      /* code */
    }else{
      rttInNs = varMap->pairRttInNs[srcNode][dstNode];
    }


    flow_entry_t flow = generate_single_rdma_flow(1, srcNode, dstNode, port, flowSize, startTime, 3, winInByte, rttInNs) ;
    RdmaClientHelper clientHelper(flow.prioGroup, flow.srcAddr, flow.dstAddr, flow.srcPort, flow.dstPort, flow.byteCnt, flow.winInByte, flow.rttInNs);
    ApplicationContainer appCon = clientHelper.Install(flow.srcNode);
    appCon.Start(Seconds(flow.startTimeInSec));
    
}






void install_kv_cache_applications(global_variable_t * varMap) {
    varMap->kvCachePara = parse_kv_cache_parameter(varMap);
    uint32_t jobNum = varMap->kvCachePara.size();

    for (uint32_t jobIdx = 0; jobIdx < jobNum; jobIdx++) {
      if (varMap->kvCachePara[jobIdx].type == KV_CACHE_INCAST) {
          iterate_single_incast_kv_cache_application(varMap, jobIdx);
      }else if (varMap->kvCachePara[jobIdx].type == KV_CACHE_BROADCAST) {
          iterate_single_broadcast_kv_cache_application(varMap, jobIdx);
      }else if (varMap->kvCachePara[jobIdx].type == KV_CACHE_INCA) {
          iterate_single_inca_kv_cache_application(varMap, jobIdx);
      }else if (varMap->kvCachePara[jobIdx].type == KV_CACHE_RING) {
          iterate_single_ring_kv_cache_application(varMap, jobIdx);
      }else{
        std::cout << "Error in install_kv_cache_applications() with wrong <jobIdx, type, jobNum>" << jobIdx << ", ";
        std::cout << varMap->kvCachePara[jobIdx].type << ", " << jobNum << std::endl;
      }
    }
}





void sim_finish(global_variable_t * varMap){
  std::cout << "111111111" << std::endl;
  if (varMap->enableQbbTrace) {
    fflush(varMap->qbbFileHandle);
    fclose(varMap->qbbFileHandle);
  }
  if (varMap->enableQlenMonitor) {
    fflush(varMap->qlenMonitorFileHandle);
    fclose(varMap->qlenMonitorFileHandle);
  }
  
  if (varMap->enableFctMonitor) {
    fflush(varMap->fctFileHandle);
    fclose(varMap->fctFileHandle);
  }
  if (varMap->enablePfcMonitor) {
    fflush(varMap->pfcFileHandle);
    fclose(varMap->pfcFileHandle);
  }
  std::cout << "222222222222" << std::endl;

  // fclose(varMap->nicInfoOutputFileHandle);

  // print_EST_to_file(varMap->paraOutputFileName, varMap->paraStrMap);
}

Ipv4Address node_id_to_ip(uint32_t id){
    return Ipv4Address(0x0b000001 + ((id / 256) * 0x00010000) + ((id % 256) * 0x00000100));
}

void parse_default_configures(global_variable_t * varMap){

  // varMap->configFileName = varMap->configFileName;
  // varMap->topoFileName = varMap->topoFileName;
  // varMap->workLoadFileName = varMap->workLoadFileName;
  // varMap->tfcFileName = varMap->tfcFileName;
  // varMap->kvCacheFileName = varMap->inputFileDir + varMap->kvCacheFileName;
  std::ifstream fh(varMap->configFileName.c_str());
  if (!fh.is_open()) {
      std::cout << "load_default_configures() 无法打开文件" << varMap->configFileName <<std::endl;
      return ;
  }

  std::vector<std::vector<std::string> > resLines;
  read_files_by_line(fh, resLines);
  fh.close();
  for (auto & e : resLines) {
    std::string key = e[0];
    std::vector<std::string> values(e.begin()+1, e.end());
    varMap->configMap[key] = values;
  }

  varMap->flowCount = 0;
  varMap->totalFlowSizeInByte = 0;
  varMap->smallFlowCount = 0;
  varMap->largeFlowCount = 0;
  varMap->flowCount = 0;
  varMap->flowCount = 0;



  varMap->pfcFileName = varMap->outputFileDir + varMap->fileIdx + "-PFC.txt";
  varMap->fctFileName = varMap->outputFileDir + varMap->fileIdx +"-FCT.txt";
  varMap->qbbFileName = varMap->outputFileDir + varMap->fileIdx +"-QBB.txt";
  varMap->qlenMonitorFileName = varMap->outputFileDir + varMap->fileIdx + "-QLEN.txt";



  varMap->pfcPauseTimeInUs = std::stoi(varMap->configMap["pfcPauseTimeInUs"][0]);
  varMap->enableQcn = integer_to_bool(varMap->configMap["enableQcn"][0]);
  varMap->enablePfcDynThresh = integer_to_bool(varMap->configMap["enablePfcDynThresh"][0]);
  varMap->intMulti = std::stoi(varMap->configMap["intMulti"][0]);
  varMap->ccMode = std::stoi(varMap->configMap["ccMode"][0]);
  varMap->pintLogBase = std::stof(varMap->configMap["pintLogBase"][0]);
  varMap->pintBytes = Pint::get_n_bytes();
  // varMap->addrFileName = varMap->configMap["addrFileName"][0];


  varMap->alphaShiftInLog = std::stoi(varMap->configMap["alphaShiftInLog"][0]);
  varMap->ecnParaMap = parse_ecn_parameter(varMap->configMap["ecnParaMap"]);
  varMap->mmuSwBufferSizeInMB = std::stoi(varMap->configMap["mmuSwBufferSizeInMB"][0]);
  varMap->nicParas.enableClampTargetRate = integer_to_bool(varMap->configMap["ClampTargetRate"][0]);
  varMap->nicParas.alphaResumeIntervalInUs = std::stof(varMap->configMap["alphaResumeIntervalInUs"][0]);
  varMap->nicParas.rpTimerInUs = std::stof(varMap->configMap["rpTimerInUs"][0]);
  varMap->nicParas.fastRecoveryTimes = std::stoi(varMap->configMap["fastRecoveryTimes"][0]);
  varMap->nicParas.ewmaGain = std::stof(varMap->configMap["ewmaGain"][0]);
  varMap->nicParas.rateAiInMbps = std::stof(varMap->configMap["rateAiInMbps"][0]);
  varMap->nicParas.rateHaiInMbps = std::stof(varMap->configMap["rateHaiInMbps"][0]);
  varMap->nicParas.enableL2BackToZero = integer_to_bool(varMap->configMap["enableL2BackToZero"][0]);
  varMap->nicParas.l2ChunkSizeInByte = std::stoi(varMap->configMap["l2ChunkSizeInByte"][0]);
  varMap->nicParas.l2AckIntervalInByte = std::stoi(varMap->configMap["l2AckIntervalInByte"][0]);
  varMap->nicParas.rateDecreaseIntervalInUs = std::stof(varMap->configMap["rateDecreaseIntervalInUs"][0]);
  varMap->nicParas.minRateInMbps = std::stoi(varMap->configMap["minRateInMbps"][0]);
  varMap->pktPayloadSizeInByte = std::stoi(varMap->configMap["pktPayloadSizeInByte"][0]);
  varMap->nicParas.miThresh = std::stoi(varMap->configMap["miThresh"][0]);
  varMap->nicParas.enableVarWin = integer_to_bool(varMap->configMap["enableVarWin"][0]);
  varMap->nicParas.enableFastFeedback = integer_to_bool(varMap->configMap["enableFastFeedback"][0]);
  varMap->nicParas.enableMultiRate = integer_to_bool(varMap->configMap["enableMultiRate"][0]);
  varMap->nicParas.enableSampleFeedback = integer_to_bool(varMap->configMap["enableSampleFeedback"][0]);
  varMap->nicParas.targetUtilization = std::stof(varMap->configMap["targetUtilization"][0]);
  varMap->nicParas.enableRateBound = integer_to_bool(varMap->configMap["enableRateBound"][0]);
  varMap->nicParas.dctcpRateAiInMbps = std::stof(varMap->configMap["dctcpRateAiInMbps"][0]);
  varMap->nicParas.pintProbTresh = std::stof(varMap->configMap["pintProbTresh"][0]);
  varMap->enableAckHigherPrio = integer_to_bool(varMap->configMap["enableAckHigherPrio"][0]);
  // varMap->traceNodeIds = parse_trace_nodes(varMap);

  varMap->loadRatioShift = std::stof(varMap->configMap["loadRatioShift"][0]);
  varMap->loadRatio = std::stof(varMap->configMap["loadRatio"][0]);
  varMap->appStartPort = std::stoi(varMap->configMap["appStartPort"][0]);
  varMap->flowGroupPrio = std::stoi(varMap->configMap["flowGroupPrio"][0]);
  varMap->enableWindow = integer_to_bool(varMap->configMap["enableWindow"][0]);
  varMap->enablePfc = integer_to_bool(varMap->configMap["enablePfc"][0]);
  varMap->enableUnifiedWindow = integer_to_bool(varMap->configMap["enableUnifiedWindow"][0]);
  // varMap->largeFLowThreshInByte = std::stoi(varMap->configMap["largeFLowThreshInByte"][0]);
  // varMap->smallFlowThreshInByte = std::stoi(varMap->configMap["smallFlowThreshInByte"][0]);
  // varMap->qlenMonitorIntervalInNs = std::stoi(varMap->configMap["qlenMonitorIntervalInNs"][0]);
  // varMap->qlenMonitorEndTimeInNs = std::stoull(varMap->configMap["qlenMonitorEndTimeInNs"][0]);
  // varMap->qlenMonitorStartTimeInNs = std::stoi(varMap->configMap["qlenMonitorStartTimeInNs"][0]);
  // varMap->flowLunchEndTimeInSec = std::stof(varMap->configMap["flowLunchEndTimeInSec"][0]);
  varMap->defaultPktSizeInByte = std::stoi(varMap->configMap["defaultPktSizeInByte"][0]);
  varMap->enableQlenMonitor = integer_to_bool(varMap->configMap["enableQlenMonitor"][0]);
  varMap->enableQbbTrace = integer_to_bool(varMap->configMap["enableQbbTrace"][0]);
  varMap->enablePfcMonitor = integer_to_bool(varMap->configMap["enablePfcMonitor"][0]);
  varMap->enableFctMonitor = integer_to_bool(varMap->configMap["enableFctMonitor"][0]);
  varMap->screenDisplayInNs = std::stoull(varMap->configMap["screenDisplayInNs"][0]);
  varMap->enableKvCache = integer_to_bool(varMap->configMap["enableKvCache"][0]);
  // varMap->incastKvCacheInfo.roundNum = std::stoi(varMap->configMap["roundNum"][0]);
  // varMap->incastKvCacheInfo.notifySizeInByte = std::stoull(varMap->configMap["notifySizeInByte"][0]);
  // varMap->incastKvCacheInfo.attentionSizeInByte = std::stoull(varMap->configMap["attentionSizeInByte"][0]);
  // varMap->incastKvCacheInfo.roundDelayInNs = std::stoull(varMap->configMap["roundDelayInNs"][0]);

}

std::map<uint32_t, ecn_para_entry_t> parse_ecn_parameter(std::vector<std::string>& s){
  std::map<uint32_t, ecn_para_entry_t> ecnParaMap;
  uint32_t len = s.size();
  if (len % ECN_PARA_NUMBER != 0) {
    std::cout <<"Error in parse_ecn_parameter() with " << std::endl;
    return ecnParaMap;
  }
  for (uint32_t i = 0; i < len/ECN_PARA_NUMBER; i++) {
    ecn_para_entry_t ecnParaEntry;
    uint32_t rateInGbps = std::stoi(s[i*ECN_PARA_NUMBER]);
    ecnParaEntry.kminInKb = std::stoi(s[i*ECN_PARA_NUMBER+1]);
    ecnParaEntry.kmaxInKb = std::stoi(s[i*ECN_PARA_NUMBER+2]);
    ecnParaEntry.pmax = std::stof(s[i*ECN_PARA_NUMBER+3]);
    ecnParaMap[rateInGbps] = ecnParaEntry;
  }
    return ecnParaMap;
}

std::map<uint32_t, kv_cache_para_t> parse_kv_cache_parameter(global_variable_t * varMap){
  std::map<uint32_t, kv_cache_para_t> kvCacheParaMap;

  std::ifstream fh(varMap->kvCacheFileName.c_str());
  if (!fh.is_open()) {
      std::cout << "parse_kv_cache_parameter()无法打开文件" << varMap->kvCacheFileName <<std::endl;
      return kvCacheParaMap;
  }

  std::vector<std::vector<std::string> > resLines;
  read_files_by_line(fh, resLines);
  fh.close();


  for (auto & e : resLines) {
    kv_cache_para_t kvPara;
    kvPara.idx = std::stoi(e[0]);
    kvPara.type = std::stoi(e[1]);
    kvPara.roundNum = std::stoi(e[2]);
    kvPara.notifySizeInByte = std::stoull(e[3]);
    kvPara.querySizeInByte = std::stoull(e[4]);
    kvPara.attentionSizeInByte = std::stoull(e[5]);
    kvPara.otherTimeInNs = std::stoull(e[6]);
    kvPara.reduceTimeInNs = std::stoull(e[7]);
    kvPara.attentionTimeInNs = std::stoull(e[8]);
    kvPara.roundCnt = 0;
    kvPara.completeCnt = 0;

    if (kvPara.type == KV_CACHE_INCA) {
      if (varMap->swNodes.GetN() == 6 && (varMap->svNodes.GetN() == 16) ) {
        kvPara.completeNum = 1;
            kvPara.followerNodes.Add(varMap->svNodes.Get(0));
            kvPara.nodeTypeMap[varMap->svNodes.Get(0)] = "FOLLOWER";

            kvPara.leaderNodes.Add(varMap->svNodes.Get(15));
            kvPara.nodeTypeMap[varMap->svNodes.Get(15)] = "LEADER";
          
        kvPara.state = 0;
        kvCacheParaMap[kvPara.idx] = kvPara;
        continue;
      }
      
      kvPara.completeNum = varMap->svNodes.GetN()/2;
        for (uint32_t i = 0; i < kvPara.completeNum; i++) {
          kvPara.followerNodes.Add(varMap->svNodes.Get(i));
          kvPara.nodeTypeMap[varMap->svNodes.Get(i)] = "FOLLOWER";
        }
        for (uint32_t i = kvPara.completeNum; i < varMap->svNodes.GetN(); i++) {
          kvPara.leaderNodes.Add(varMap->svNodes.Get(i));
          kvPara.nodeTypeMap[varMap->svNodes.Get(i)] = "LEADER";
        }
      kvPara.state = 0;
      kvCacheParaMap[kvPara.idx] = kvPara;
      continue;
    }else if (kvPara.type == KV_CACHE_RING) {
      kvPara.completeNum = varMap->svNodes.GetN();
        for (uint32_t i = 0; i < kvPara.completeNum; i++) {
          kvPara.followerNodes.Add(varMap->svNodes.Get(i));
          kvPara.leaderNodes.Add(varMap->svNodes.Get((i+1)%kvPara.completeNum));
        }
      kvCacheParaMap[kvPara.idx] = kvPara;
      continue;
    }else if (kvPara.type == KV_CACHE_INCAST)  {
      uint32_t followerNum = varMap->svNodes.GetN()-1;
      for (uint32_t i = 0; i < followerNum; i++) {
        kvPara.followerNodes.Add(varMap->svNodes.Get(i));
      }
      kvPara.leaderNode = varMap->svNodes.Get(followerNum);
      kvPara.completeNum = followerNum;
      kvCacheParaMap[kvPara.idx] = kvPara;
      continue;
    }
    
    
    
    uint32_t followerNum = varMap->svNodes.GetN()-1;
    for (uint32_t i = 0; i < followerNum; i++) {
      kvPara.followerNodes.Add(varMap->svNodes.Get(i));
    }
    kvPara.leaderNode = varMap->svNodes.Get(followerNum);
    kvPara.completeNum = followerNum;
    kvCacheParaMap[kvPara.idx] = kvPara;
  }
    return kvCacheParaMap;
}

std::vector<uint32_t> parse_trace_nodes(global_variable_t * varMap){
  std::vector<std::string> nodesIdxes = varMap->configMap["traceNodeIds"];
  std::vector<uint32_t> traceNodes;
  for (uint32_t i = 0; i < nodesIdxes.size(); i++) {
    uint32_t nodeIdx = std::stoi(nodesIdxes[i]);
    traceNodes.push_back(nodeIdx);
  }
    return traceNodes;
}

void load_default_configures(global_variable_t * varMap){

    Config::SetDefault("ns3::QbbNetDevice::PauseTime", UintegerValue(varMap->pfcPauseTimeInUs));
    Config::SetDefault("ns3::QbbNetDevice::QcnEnabled", BooleanValue(varMap->enableQcn));
    Config::SetDefault("ns3::QbbNetDevice::DynamicThreshold", BooleanValue(varMap->enablePfcDynThresh));
    Config::SetDefault("ns3::QbbNetDevice::QbbEnabled", BooleanValue(varMap->enablePfc));
    Config::SetDefault("ns3::SwitchNode::LbSolution", StringValue ("ecmp"));

    IntHop::multi = varMap->intMulti;
    if (varMap->ccMode == 3) { // hpcc
      IntHeader::mode = IntHeader::NORMAL;
    }else if (varMap->ccMode == 7) { // timely
        IntHeader::mode = IntHeader::TS;
    }else if (varMap->ccMode == 10) { //pint
        IntHeader::mode = IntHeader::PINT;
    }else{
        IntHeader::mode = IntHeader::NONE;
    }
    
    
    
    // IntHeader::mode = IntHeader::Mode(varMap->ccMode);
    Pint::set_log_base(varMap->pintLogBase);
    IntHeader::pint_bytes = varMap->pintBytes;
    // varMap->cdfTable = new cdf_table();
    // init_cdf(varMap->cdfTable);
    varMap->qlenMonitorFileHandle = NULL;
    varMap->numOfFinishedJob = 0;
    Simulator::Schedule(NanoSeconds(0), &screen_display, varMap->screenDisplayInNs);

}


// struct reorderDistEntry {
//   std::vector<uint32_t> freq;
//   uint64_t maxValue;
//   uint64_t maxIndex;
//   uint32_t counter;
//   reorderDistEntry(){
//     freq.clear();
//     maxIndex = 0;
//     maxValue = 0;
//     counter = 0;
//   }
//   bool operator==(const reorderDistEntry& other) const {
//     return maxValue == other.maxValue && maxIndex == other.maxIndex && counter == other.counter;
//   }
//   reorderDistEntry(uint32_t seq){
//     freq.clear();
//     maxIndex = 1;
//     maxValue = seq;
//     counter = 1;
//   }
//   void Print(std::ostream &os){
//     os << vector2string<uint32_t> (freq, ", ") << std::endl;
//   }
// };

void SinkRx (Ptr<const Packet> p, const Address &from, const Address & local, const SeqTsSizeHeader& hdr) {
  std::cout << "<srcIP=" << InetSocketAddress::ConvertFrom(from).GetIpv4 () << ", ";
  std::cout << "sPort=" << InetSocketAddress::ConvertFrom(from).GetPort () << ">, ";
  std::cout << "<dstIP=" << InetSocketAddress::ConvertFrom(local).GetIpv4 () << ", ";
  std::cout << "dPort=" << InetSocketAddress::ConvertFrom(local).GetPort () << ">, ";
  std::cout << "pktSize= " << hdr.GetSize() << ", ";
  std::cout << "SeqNum= " << hdr.GetSeq() << "\n";

  // Ptr<OutputStreamWrapper> screenStream = Create<OutputStreamWrapper> (&std::cout);
  // hdr.Print(std::cout);
  // std::cout << "\n";
  // std::cout << "Packet: " << (unsigned)ipv4.GetTtl () << std::endl;
  // TcpHeader tcpHeader;
  // uint32_t bytesRemoved = p->PeekHeader (tcpHeader);
  // std::cout << "SeqNumer: " << tcpHeader.GetSequenceNumber () << std::endl;

}


// void tcpSocketBaseBxCb (std::map<std::string, reorderDistEntry> * reorderDistTbl, std::string fid, Ptr<const Packet> p, const TcpHeader& tcpHdr,  Ptr<const TcpSocketBase> skt) {
//       Address from, to;
//       skt->GetSockName(to);
//       Ipv4Address dstIpv4Addr = InetSocketAddress::ConvertFrom(to).GetIpv4 ();
//       uint16_t dstPort = InetSocketAddress::ConvertFrom(to).GetPort ();
//       skt->GetPeerName(from);
//       Ipv4Address srcIpv4Addr = InetSocketAddress::ConvertFrom(from).GetIpv4 ();
//       uint16_t srcPort = InetSocketAddress::ConvertFrom(from).GetPort ();
//       uint32_t seq = tcpHdr.GetSequenceNumber ().GetValue();
//       uint32_t pktSieInByte = p->GetSize();
//       // std::cout << "Seq: " << seq << ", PacketSize: " << pktSieInByte << std::endl;

//       auto it = (*reorderDistTbl).find(fid);
//       if (it == (*reorderDistTbl).end()) {
//         reorderDistEntry * e = new reorderDistEntry(seq, pktSieInByte);
//         (*reorderDistTbl)[fid] = *e;
//       }else{
//         it->second.counter += 1;
//         it->second.size += pktSieInByte;
//         it->second.lastTime = Now().GetNanoSeconds();
//         if (seq >= it->second.maxValue) {
//           it->second.maxIndex = it->second.counter;
//           it->second.maxValue = seq;
//         }else{
//           uint32_t d = (it->second.maxValue-1-seq)/DEFAULT_MAX_TCP_MSS_IN_BYTE + 1;
//           if (it->second.freq.size() < d) {
//             it->second.freq.resize(d, 0);
//           }
//           it->second.freq[d-1] += 1;
//         }
//       }
// }






// void PrintNodeQlen(FILE * os, NodeContainer nodes, uint64_t interval, uint32_t round){
//     // FILE * os = fopen(file.c_str(), "w");
//     // if (os == NULL) {
//     //   std::cout << "Error for Cannot open file " << file << std::endl;
//     // }
    
//     uint64_t curTimeInNs = Now().GetNanoSeconds();
//     for (uint32_t i = 0; i < nodes.GetN(); i++){
//       Ptr<Node> tmpNode = nodes.Get(i);
//       std::string tmpNodeName = GetNodeName (tmpNode, true);
//       Ptr<Ipv4DeflowRouting> rp = DynamicCast<Ipv4DeflowRouting> (tmpNode->GetObject<Ipv4>()->GetRoutingProtocol());
//       if (!rp) {
//         continue;
//       }
//       uint32_t n_nic = tmpNode->GetNDevices();
//       if (n_nic <= 2) {
//         continue;
//       }
//       std::vector<uint32_t> qlen(n_nic-1, 0);
//       for (uint32_t j = 1; j < n_nic; j++){
//           qlen[j-1] = rp->GetQueueLength(j);
//       }
//       std::string qlenStr = vector2string<uint32_t> (qlen, ", ");
//       fprintf(os, "r:%d t:%lu n:%s q:%s\n", round, curTimeInNs, tmpNodeName.c_str(), qlenStr.c_str());
//     }
//     fflush(os);

//     if ((curTimeInNs + interval) < Simulator::GetMaximumSimulationTime().GetNanoSeconds()){
//         Simulator::Schedule(NanoSeconds(interval), &PrintNodeQlen, os, nodes, interval, round+1);
//     }
// }

void PrintNodeQlen(std::vector<std::vector<QueueLengthEntry> > * qlenTbl, NodeContainer nodes, uint64_t interval, uint32_t round){
      const uint32_t n_node = nodes.GetN();
    
      if (round == 1) {
        qlenTbl->resize(n_node);
        for (uint32_t i = 0; i < n_node; i++){
          Ptr<Node> tmpNode = nodes.Get(i);
          const uint32_t n_nic = tmpNode->GetNDevices();
          (*qlenTbl)[i].resize(n_nic-1);
        }
      }
  

    for (uint32_t i = 0; i < n_node; i++){
      Ptr<Node> tmpNode = nodes.Get(i);
      Ptr<Ipv4DeflowRouting> rp = DynamicCast<Ipv4DeflowRouting> (tmpNode->GetObject<Ipv4>()->GetRoutingProtocol());
      if (!rp) {
        continue;
      }
      const uint32_t n_nic = tmpNode->GetNDevices();
      if (n_nic <= 2) {
        continue;
      }
      for (uint32_t j = 1; j < n_nic; j++){
          uint32_t tmpLen = rp->GetQueueLength(j);
          if (round == 1) {
            QueueLengthEntry * tmpEntry = new QueueLengthEntry(tmpLen);
            (*qlenTbl)[i][j-1] = *tmpEntry;
          }else{
            (*qlenTbl)[i][j-1].round = round;
            (*qlenTbl)[i][j-1].maxLen = std::max((*qlenTbl)[i][j-1].maxLen, tmpLen);
            (*qlenTbl)[i][j-1].totalLen += tmpLen;

            uint32_t lastLen = (*qlenTbl)[i][j-1].len.back().second;
            if (tmpLen != lastLen) {
              (*qlenTbl)[i][j-1].len.push_back(std::make_pair(round, tmpLen));
            }            
          } 
          // (*qlenTbl)[i][j-1].len.push_back(tmpLen);
          // (*qlenTbl)[i][j-1].maxLen = std::max((*qlenTbl)[i][j-1].maxLen, tmpLen);
      }
    }


    
    uint64_t curTimeInNs = Now().GetNanoSeconds();
    if (((curTimeInNs + interval) < Simulator::GetMaximumSimulationTime().GetNanoSeconds())&&(!Simulator::IsFinished())){
        Simulator::Schedule(NanoSeconds(interval), &PrintNodeQlen, qlenTbl, nodes, interval, round+1);
    }
}


void SaveNodeQlen(std::string file, std::vector<std::vector<QueueLengthEntry> > & qlenTbl, uint64_t intervalInNs){
  std::ofstream os(file.c_str());
  if (!os.is_open()) {
    std::cout << "SaveNodeQlen() cannot open file " << file << std::endl;
    return ;
  }
  const uint32_t n_node = qlenTbl.size();
  os << n_node << " " << intervalInNs << std::endl;
  for (uint32_t i = 0; i < n_node; i++) {
    const uint32_t n_nic = qlenTbl[i].size();
    if (n_nic <= 1) {
      continue;
    }
    
    os << std::setiosflags (std::ios::left) << std::setw (3) << i;
    os << std::setiosflags (std::ios::left) << std::setw (3) << n_nic << "\n";
    for (uint32_t j = 0; j < n_nic; j++) {
      os << std::setiosflags (std::ios::left) << std::setw (3) << j+1 << " ";
      os << qlenTbl[i][j].ToString();
      os << "\n";
    }
  }
  os.close();
}

void DisplayNodeQlen(std::vector<std::vector<QueueLengthEntry> > & qlenTbl, uint64_t intervalInNs){
  const uint32_t n_node = qlenTbl.size();

  std::cout << "******************************************************";
  std::cout << "Qlen Tables of " << n_node << " nodes with period of " << intervalInNs/1000 << " us";
  std::cout << "******************************************************";
  std::cout << "\n";

  std::cout << std::setiosflags (std::ios::left) << std::setw (5) << "Node";
  std::cout << std::setiosflags (std::ios::left) << std::setw (5) << "#Nic";
  std::cout << std::setiosflags (std::ios::left) << std::setw (5) << "Nic";
  std::cout << std::setiosflags (std::ios::left) << std::setw (10) << "#Round";
  std::cout << std::setiosflags (std::ios::left) << std::setw (10) << "AvgLen";
  std::cout << std::setiosflags (std::ios::left) << std::setw (10) << "MaxLen";
  std::cout << std::setiosflags (std::ios::left) << std::setw (10) << "Details";
  std::cout << "\n";

  for (uint32_t i = 0; i < n_node; i++) {
    const uint32_t n_nic = qlenTbl[i].size();
    if (n_nic <= 1) {
      continue;
    }
    for (uint32_t j = 0; j < n_nic; j++) {
      std::cout << std::setiosflags (std::ios::left) << std::setw (5) << i;
      std::cout << std::setiosflags (std::ios::left) << std::setw (5) << n_nic;
      std::cout << std::setiosflags (std::ios::left) << std::setw (5) << j+1;
      std::cout << qlenTbl[i][j].ToString();
      std::cout << "\n";
    }
  }
}




void SaveReorderDregree(std::string file, std::map<std::string, reorderDistEntry> & reorderDistTbl){
  std::ofstream os(file.c_str());
  if (!os.is_open()) {
    std::cout << "SaveReorderDregree() cannot open file " << file << std::endl;
    return ;
  }
  uint64_t initialValue = 0;
  std::vector<std::pair<std::string, reorderDistEntry> > sortedVec(reorderDistTbl.begin(), reorderDistTbl.end());
  auto compareReorderDistEntryByDecreasingSize =  [](const std::pair<std::string, reorderDistEntry>& a,  const std::pair<std::string, reorderDistEntry>& b) {
          return a.second.size > b.second.size; // 降序排序
  };
  std::sort(sortedVec.begin(), sortedVec.end(), compareReorderDistEntryByDecreasingSize);
  const uint32_t n_flow = sortedVec.size();
  auto accumulateBySize = [](uint64_t acc, const std::pair<std::string, reorderDistEntry>& s) {
    return acc + s.second.size;
  };
  uint64_t bytes = std::accumulate(sortedVec.begin(), sortedVec.end(), initialValue, accumulateBySize);
  os << n_flow << " " << bytes << std::endl;
  uint32_t c_flow = 1;
  for (auto & it : sortedVec) {
    os << std::setiosflags (std::ios::left) << std::setw (7) << c_flow++;
    os << std::setiosflags (std::ios::left) << std::setw (45) << it.first;
    os << it.second.ToString();
    os << "\n";
  }
  os.close();
}

std::vector<CommunicationPatternEntry> read_pattern_file(std::string trafficFile) {
  std::vector<CommunicationPatternEntry> pEntries;
  // std::srand(static_cast<unsigned int>(std::time(0)));  
  std::ifstream fh_trafficFile(trafficFile.c_str());
  if (!fh_trafficFile.is_open()) {
        std::cerr << "read_pattern_file() 无法打开文件 " << trafficFile << std::endl;
        return pEntries;
  }
  std::vector<std::vector<std::string> > resLines;
  uint32_t lineCnt = read_files_by_line(fh_trafficFile, resLines);
  fh_trafficFile.close();
  std::cout << "read_pattern_file() reads file: " << trafficFile << " about " << lineCnt << " Lines" <<std::endl;

  for (uint32_t i = 0; i < lineCnt; i++) {
    if (resLines[i].size() < 3) {
        std::cerr << "read_pattern_file() 输入错误 " << resLines[i].size() << std::endl;
        return pEntries;
    }
    CommunicationPatternEntry pEntry(resLines[i][0], resLines[i][1], std::atof(resLines[i][2].c_str()));
    pEntries.push_back(pEntry);
  }
  return pEntries;
}


struct cdf_table * read_workload_file(std::string file) {
    struct cdf_table * cdfTable = new cdf_table();
    init_cdf(cdfTable);
    load_cdf(cdfTable, file.c_str());
    return cdfTable;
}

void install_tcp_bulk_apps( std::vector<CommunicationPatternEntry>& pEntries, struct cdf_table *cdfTable,
                            double START_TIME, double END_TIME, double FLOW_LAUNCH_END_TIME,
                            uint16_t startAppPort, double loadFactor) {

    long totalFlowSize = 0, totalFlowCount = 0;
    uint32_t tmpSmallFlowCount=0, tmpLargeFlowCount=0;

    for (auto & pEntry : pEntries) {
        Ptr<Node> srcNode = Names::Find<Node> (pEntry.srcNodeName);
        Ptr<Node> dstNode = Names::Find<Node> (pEntry.dstNodeName);
        if (srcNode == 0 || dstNode == 0)  {
          std::cerr << "install_tcp_bulk_apps() 输入错误: " << pEntry.ToString() << std::endl;
          return ;
        }
        double tmpLoadFactor = pEntry.loadAdjustRate * loadFactor;
        Ptr<PointToPointNetDevice> p2pDev = DynamicCast<PointToPointNetDevice> (dstNode->GetObject<Ipv4> ()->GetNetDevice (1));
        if (p2pDev == 0) {
          std::cerr << "install_tcp_bulk_apps() 输入错误: " << pEntry.ToString() << std::endl;
          return ;      
        }
        uint64_t bitRate = p2pDev->GetDataRate().GetBitRate();
        double requestRate = tmpLoadFactor * bitRate / (8 * avg_cdf (cdfTable));
        install_flows_in_tcp_bulk_on_node_pair( srcNode, dstNode, requestRate, cdfTable, totalFlowCount, totalFlowSize,
                                                START_TIME, END_TIME, FLOW_LAUNCH_END_TIME,
                                                startAppPort, tmpSmallFlowCount, tmpLargeFlowCount);
    }
    std::cout << "FlowCount: " << totalFlowCount << std::endl;
    std::cout << "FlowSize: " << totalFlowSize << std::endl;
    std::cout << "SmallFlowCount: " << tmpSmallFlowCount << std::endl;
    std::cout << "LargeFlowCount: " << tmpLargeFlowCount << std::endl;
    std::cout << "#Communication Pair: " << pEntries.size() << std::endl;
    return ;
}

void SaveFlowInfo(std::string file, std::map<std::string, flowInfo> & flowTable){
  std::ofstream os(file.c_str());
  if (!os.is_open()) {
    std::cout << "SaveFlowInfo() cannot open file " << file << std::endl;
    return ;
  }

  uint64_t initialValue = 0;
  std::vector<std::pair<std::string, flowInfo> > sortedVec(flowTable.begin(), flowTable.end());
  const uint32_t n_flow = sortedVec.size();

  auto compareFlowInfoByIncreasingFct =  [](const std::pair<std::string, flowInfo>& a,  const std::pair<std::string, flowInfo>& b) {
          return (a.second.completeTimeInNs-a.second.startTimeInNs) < (b.second.completeTimeInNs-b.second.startTimeInNs); // 升序排序
  };
  std::sort(sortedVec.begin(), sortedVec.end(), compareFlowInfoByIncreasingFct);
  auto accumulateByFct = [](uint64_t acc, const std::pair<std::string, flowInfo>& s) {
    // std::cout << "acc: " << acc << "\n";
    // std::cout << "s.second.completeTimeInNs: " << s.second.completeTimeInNs << "\n";
    // std::cout << "s.second.startTimeInNs: " << s.second.startTimeInNs << "\n";
    // std::cout << "gap: " << s.second.completeTimeInNs-s.second.startTimeInNs << "\n";
    // std::cout << "acc: " << acc + (s.second.completeTimeInNs-s.second.startTimeInNs) << "\n";
    return acc + (s.second.completeTimeInNs-s.second.startTimeInNs);
  };
  uint64_t totalFctInNs = std::accumulate(sortedVec.begin(), sortedVec.end(), initialValue, accumulateByFct);
  double avgFctInNs = 1.0*totalFctInNs/n_flow;
  os << std::setiosflags (std::ios::left) << std::setw (12) << n_flow;
  os << std::setiosflags (std::ios::left) << std::setw (12) << std::fixed << std::setprecision(1) << avgFctInNs;
  os << "\n";

  uint32_t smallFlowIndex = n_flow * SMALL_FLOW_RATIO;
  uint64_t smallFctInNs = std::accumulate(sortedVec.begin(), sortedVec.begin()+smallFlowIndex+1, initialValue, accumulateByFct);
  double avgSmallFctInNs = 1.0*smallFctInNs/(smallFlowIndex+1);
  os << std::setiosflags (std::ios::left) << std::setw (12) << smallFlowIndex+1;
  os << std::setiosflags (std::ios::left) << std::setw (12) << std::fixed << std::setprecision(1) << avgSmallFctInNs;
  os << "\n";

  uint32_t smallFlowIndex_99 = smallFlowIndex * 0.99;
  double smallFctInNs_99 =  sortedVec[smallFlowIndex_99].second.completeTimeInNs-sortedVec[smallFlowIndex_99].second.startTimeInNs;
  os << std::setiosflags (std::ios::left) << std::setw (12) << smallFlowIndex_99;
  os << std::setiosflags (std::ios::left) << std::setw (12) << std::fixed << std::setprecision(1) << smallFctInNs_99;
  os << "\n";

  uint32_t largeFlowIndex = std::min(n_flow-1, uint32_t(n_flow * (1-LARGE_FLOW_RATIO)));
  uint64_t largeFctInNs = std::accumulate(sortedVec.begin()+largeFlowIndex, sortedVec.end(), initialValue, accumulateByFct);
  // std::cout << "totalLargeFctInNs: " << largeFctInNs << std::endl;
  double avglargeFctInNs = 1.0*largeFctInNs/(n_flow-largeFlowIndex);
  // std::cout << "n_flow-largeFlowIndex: " << n_flow-largeFlowIndex << std::endl;
  // std::cout << "avglargeFctInNs: " << avglargeFctInNs << std::endl;

  os << std::setiosflags (std::ios::left) << std::setw (12) << n_flow-largeFlowIndex;
  os << std::setiosflags (std::ios::left) << std::setw (20) << std::fixed << std::setprecision(1) << avglargeFctInNs;
  os << "\n";

  uint32_t largeFlowIndex_99 = largeFlowIndex * 0.99;
  double largeFctInNs_99 =  sortedVec[largeFlowIndex_99].second.completeTimeInNs-sortedVec[largeFlowIndex_99].second.startTimeInNs;
  os << std::setiosflags (std::ios::left) << std::setw (12) << largeFlowIndex_99;
  os << std::setiosflags (std::ios::left) << std::setw (12) << std::fixed << std::setprecision(1) << largeFctInNs_99;
  os << "\n";


  auto compareFlowInfoByDecreasingSize =  [](const std::pair<std::string, flowInfo>& a,  const std::pair<std::string, flowInfo>& b) {
          return a.second.recvBytes > b.second.recvBytes; // 降序排序
  };
  std::sort(sortedVec.begin(), sortedVec.end(), compareFlowInfoByDecreasingSize);

  auto accumulateBySize = [](uint64_t acc, const std::pair<std::string, flowInfo>& s) {
    return acc + s.second.recvBytes;
  };
  uint64_t bytes = std::accumulate(sortedVec.begin(), sortedVec.end(), initialValue, accumulateBySize);
  uint64_t cumsum_bytes = 0;

  os << n_flow << " " << bytes << std::endl;

  for (uint32_t i = 0; i < n_flow; i++) {
    cumsum_bytes += sortedVec[i].second.recvBytes;
    os << std::setiosflags (std::ios::left) << std::setw (10) << i+1;
    os << std::setiosflags (std::ios::left) << std::setw (7) << std::fixed << std::setprecision(4) << 1.0*(i+1)/n_flow;
    os << std::setiosflags (std::ios::left) << std::setw (25) << sortedVec[i].first;
    os << std::setiosflags (std::ios::left) << std::setw (7) << std::fixed << std::setprecision(4) << 1.0*sortedVec[i].second.recvBytes/bytes;
    os << std::setiosflags (std::ios::left) << std::setw (7) << std::fixed << std::setprecision(4) << 1.0*cumsum_bytes/bytes;
    os << sortedVec[i].second.ToString();
    os << "\n";
  }

  os.close();

}



}

