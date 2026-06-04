/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
* This program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License version 2 as
* published by the Free Software Foundation;
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#undef PGO_TRAINING
#define PATH_TO_PGO_CONFIG "path_to_pgo_config"

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <time.h> 
#include "ns3/core-module.h"
#include "ns3/qbb-helper.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/applications-module.h"
#include "ns3/internet-module.h"
#include "ns3/global-route-manager.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/packet.h"
#include "ns3/error-model.h"
#include <ns3/rdma.h>
#include <ns3/rdma-client.h>
#include <ns3/rdma-client-helper.h>
#include <ns3/rdma-driver.h>
#include <ns3/switch-node.h>
#include <ns3/sim-setting.h>
#include <ns3/userdefinedfunction.h>

using namespace ns3;
using namespace std;

NS_LOG_COMPONENT_DEFINE("CongestionControlSimulator");

int main(int argc, char *argv[]) {
    LogComponentEnable ("userdefinedfunction", LOG_LEVEL_INFO);
    LogComponentEnable ("CongestionControlSimulator", LOG_LEVEL_INFO);
    LogComponentEnable ("QbbNetDevice", LOG_LEVEL_INFO);
    // LogComponentEnable ("BEgressQueue", LOG_LEVEL_INFO);

    // LogComponentEnable ("SwitchNode", LOG_LEVEL_INFO);


    

    global_variable_t varMap;
    // Simulator::Schedule(NanoSeconds(0), &screen_display, 10000);

    std::cout<<"*******************************Parse the Input Parameters*****************************************"<<std::endl;
    CommandLine cmd;
    cmd.AddValue("configFileName", "configFileName", varMap.configFileName);
    cmd.AddValue("topoFileName", "topoFileName", varMap.topoFileName);
    cmd.AddValue("kvCacheFileName", "kvCacheFileName", varMap.kvCacheFileName);
    cmd.AddValue("inputFileDir", "input File Directory", varMap.inputFileDir);
    cmd.AddValue("outputFileDir", "output File Directory", varMap.outputFileDir);
    cmd.AddValue("fileIdx", "fileIdx", varMap.fileIdx);
    cmd.AddValue("simStartTimeInSec", "simStartTimeInSec", varMap.simStartTimeInSec);
    cmd.AddValue("simEndTimeInSec", "simEndTimeInSec", varMap.simEndTimeInSec);
    cmd.AddValue("qlenMonitorIntervalInNs", "qlenMonitorIntervalInNs", varMap.qlenMonitorIntervalInNs);

    cmd.Parse (argc, argv);
    std::cout<<"*******************************Parse the Default Configures*****************************************"<<std::endl;
    parse_default_configures(&varMap);
    std::cout<<"*******************************Load the Default Configures*****************************************"<<std::endl;
    load_default_configures(&varMap);
    std::cout<<"-------------------------------Create The Topology----------------------------------------"<<std::endl;
    create_topology_rdma(&varMap);
    std::cout<<"-------------------------------Assign The Addresses----------------------------------------"<<std::endl;
    assign_addresses(varMap.allNodes, varMap.addr2node);
    std::cout<<"-------------------------------Calculate The Paths----------------------------------------"<<std::endl;
    calculate_paths_for_servers(&varMap);
    std::cout<<"-------------------------------Install The Routing Table----------------------------------------"<<std::endl;
    install_routing_entries(&varMap);
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
    std::cout<<"-------------------------------Configure The Switch----------------------------------------"<<std::endl;
    config_switch(&varMap);
    std::cout<<"-------------------------------Install The Application----------------------------------------"<<std::endl;
    install_kv_cache_applications(&varMap);
    std::cout<<"-------------------------------Monitor The queue Len----------------------------------------"<<std::endl;
    monitor_special_port_qlen(&varMap, 0, 16, 0);
    std::cout<<"-------------------------------Monitor The qBB Device----------------------------------------"<<std::endl;
    set_QBB_trace(&varMap);
    std::cout<<"-------------------------------Start the Simulation----------------------------------------"<<std::endl;
    Simulator::Stop(Seconds(varMap.simEndTimeInSec));
    NS_LOG_INFO("Run Simulation.");
    Simulator::Run();
    sim_finish(&varMap);
    Simulator::Destroy();
    std::cout<<"-------------------------------Finish The Simulation----------------------------------------"<<std::endl;
    return 0;

}
