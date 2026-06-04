import os
import sys
import argparse
import shutil

tfcFileDict={"A00016":"TFC-S.txt",\
            "A00017":"TFC-R.txt",\
            "A00018":"TFC-A.txt",\
            "A00019":"TFC-S.txt",\
            "A00020":"TFC-R.txt",\
            "A00021":"TFC-A.txt",\
            "A00022":"TFC-S.txt",\
            "A00023":"TFC-R.txt",\
            "A00024":"TFC-A.txt",\
            "A00026":"TFC-R.txt",\
            "A00049":"TFC-R.txt"}

workloadFileDict={  "A00016":"DCTCP_CDF.txt",\
                    "A00017":"DCTCP_CDF.txt",\
                    "A00018":"DCTCP_CDF.txt",\
                    "A00019":"RPC_CDF.txt",\
                    "A00020":"RPC_CDF.txt",\
                    "A00021":"RPC_CDF.txt",\
                    "A00022":"VL2_CDF.txt",\
                    "A00023":"VL2_CDF.txt",\
                    "A00024":"VL2_CDF.txt",\
                    "A00026":"VL2_CDF.txt",\
                    "A00049":"VL2_CDF.txt"}


current_file_path = os.path.abspath(__file__)
current_dir_path = os.path.dirname(current_file_path)
current_dir_name = os.path.basename(current_dir_path)
experimentalName = current_dir_name
mainFileName   =  "main"

parser = argparse.ArgumentParser(description="请输入参数：(1)srcNodeIdx (2)dstNodeIdx (3)sendingPktNum (4)sendingIntervalInNs (5)pathSelStrategy (6)pathExpiredTimeThldInNs")
parser.add_argument("--srcNodeIdx", default="36", help="测试时发送端节点的序号, 默认值36为第1台服务器")
parser.add_argument("--dstNodeIdx", default="39", help="测试时接收端节点的序号, 默认值39为第3台服务器")
parser.add_argument("--sendingPktNum", default="1", help="测试时发送报文数量, 默认值2")
parser.add_argument("--sendingIntervalInNs", default="1000000", help="测试时发送报文时间间隔(单位：纳秒), 默认值1-000-000,1毫秒")

args = parser.parse_args()


vm_root_path = "/file-in-ctr/"
if not os.path.exists(vm_root_path):
    vm_root_path = "/file-in-cntr/"

vm_inputFiles_path = vm_root_path + "inputFiles/" + experimentalName + "/"
vm_outputFiles_path = vm_root_path + "outputFiles/" + experimentalName + "/"
vm_executable_path = vm_root_path + "executableFiles/" + experimentalName + "/"
vm_mainFile_path = vm_executable_path + mainFileName + ".cc"

vm_workload_path = vm_root_path + "workLoad/"
vm_userdefinedfunction_path = vm_root_path + "userdefinedfunction/"
vm_smartflow_path = vm_root_path + "smartflow-routing/"
vm_userdefinedfunction_model_path = vm_userdefinedfunction_path + "model/"
vm_smartflow_model_path = vm_smartflow_path + "model/"

ns3_base_path = "/app/ns3-detnet-rdma-main/ns-3.33/"
ns3_smartflow_path = ns3_base_path + "src/smartflow-routing/"
ns3_userdefinedfunction_path = ns3_base_path + "src/userdefinedfunction/"
ns3_scratch_path = "/app/ns3-detnet-rdma-main/ns-3.33/scratch/"
ns3_mainFile_path = ns3_scratch_path + mainFileName + ".cc"
ns3_userdefinedfunction_model_path = ns3_userdefinedfunction_path + "model/"
ns3_smartflow_model_path = ns3_smartflow_path + "model/"

ns3_waf_path = "/app/ns3-detnet-rdma-main/ns-3.33/"


# update the main.cc
if os.path.exists(ns3_mainFile_path):
    os.remove(ns3_mainFile_path)
shutil.copy(vm_mainFile_path, ns3_scratch_path)

# update the userdefinedfunction/model/
if os.path.exists(ns3_userdefinedfunction_model_path):
    shutil.rmtree(ns3_userdefinedfunction_model_path)   #递归删除文件夹下的所有子文件夹和子文件和其本身
shutil.copytree(vm_userdefinedfunction_model_path, ns3_userdefinedfunction_model_path)

# update the smartflow-routing/model/
if os.path.exists(ns3_smartflow_model_path):
    shutil.rmtree(ns3_smartflow_model_path)   #递归删除文件夹下的所有子文件夹和子文件
shutil.copytree(vm_smartflow_model_path, ns3_smartflow_model_path)

# create the sub dir for the results in outputFiles/
if os.path.exists(vm_outputFiles_path):
    shutil.rmtree(vm_outputFiles_path)   #递归删除文件夹下的所有子文件夹和子文件
os.makedirs(vm_outputFiles_path)

# specify the input files
addrFile      = vm_inputFiles_path + "ADDR.txt" 
vmtFile       = vm_inputFiles_path + "VMT.txt" 
pstFile       = vm_inputFiles_path + "PST.txt" 
pitFile       = vm_inputFiles_path + "PIT.txt" 
tfcFile       = vm_inputFiles_path + tfcFileDict[experimentalName]
topoFile      = vm_inputFiles_path + "TOPO.txt" 
chlFile       = vm_inputFiles_path + "CHL.txt" 
workloadFile  = vm_workload_path   + workloadFileDict[experimentalName]
# specify the outputput files



os.chdir(ns3_waf_path)

reTxThreshold = 5
simStartInSec = 0.00
simEndInSec = 0.1
flowLaunchEndInSec = 0.1
pathExpiredTimeThldInNs = 110000 #110us
probeStrategy = 1 #"{0:smallLentcyFirst}, {1: older First}, {2:random}"
piggyLatencyCnt = 2
piggybackStrategy = 2 #{0:LtyFirst},  {1:freshFirst}, {2:SntFirst}, 默认值0
flowletTimeoutInNs = 100000 #100us
probeTimeIntervalInNs = 110000 #110us

# ListOfLoadFactor = ['0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95','1.0']
# ListOfLoadFactor = ['0.5', '0.6', '0.7', '0.8',  '0.9', '1.0']

# ListOfLoadFactor = ['0.5', '0.55']

ListOfLoadFactor = ['1.0']
pathSelNum = 8
piggyLatencyCnt = 8

# ListOfPathSelStrategy = ['4', '2', '3', '5', '6'] # {1:min}, {2:rbn}, {3:rnd}, {4:lspray}, {5:flet}, {6:ecmp}, 默认值4
ListOfPathSelStrategy = ['10'] # {1:min}, {2:rbn}, {3:rnd}, {4:lspray}, {5:flet}, {6:ecmp}, 默认值4

algNameDict = {'1':'min', '2':'rbnp', '3':'rnd', '4':'lspray', '5':'flet', '6':'ecmp', '7':'rbnf', '8':'rbnh','9':'lsprayr','10':'conga'}
loadfactorDict = {'0.5':'50', '0.55':'55','0.6':'60','0.65':'65', '0.7':'70', '0.75':'75','0.8':'80', '0.85':'85','0.9':'90', '0.95':'95','1.0':'100'}

for pathSelStrategy in ListOfPathSelStrategy:
    for loadFactor in ListOfLoadFactor:
        monitorFile   = vm_outputFiles_path + algNameDict[pathSelStrategy] +  "-" + loadfactorDict[loadFactor] +  "-" + "flows.xml"
        parameterFile = vm_outputFiles_path + algNameDict[pathSelStrategy] +  "-" + loadfactorDict[loadFactor] +  "-" + "paras.txt"
        probeFile     = vm_outputFiles_path + algNameDict[pathSelStrategy] +  "-" + loadfactorDict[loadFactor] +  "-" + "probe.txt"
        reorderFile     = vm_outputFiles_path + algNameDict[pathSelStrategy] +  "-" + loadfactorDict[loadFactor] +  "-" + "reorder.txt"

        Line_command = '\
                ./waf --run "scratch/{}\
                --addrFile={} \
                --vmtFile={} \
                --pstFile={} \
                --pitFile={} \
                --tfcFile={} \
                --topoFile={} \
                --chlFile={} \
                --workloadFile={} \
                --monitorFile={} \
                --parameterFile={} \
                --probeFile={}\
                --reorderFile={}\
                --loadFactor={}\
                --simStartInSec={} \
                --simEndInSec={} \
                --flowLaunchEndInSec={} \
                --pathSelStrategy={} \
                --pathExpiredTimeThldInNs={} \
                --probeStrategy={} \
                --piggyLatencyCnt={} \
                --piggybackStrategy={} \
                --flowletTimeoutInNs={} \
                --pathSelNum={}\
                --reTxThreshold={}\
                --probeTimeIntervalInNs={} \
                --srcNodeIdx={} \
                --dstNodeIdx={} \
                --sendingPktNum={} \
                --sendingIntervalInNs={} \
                "\
                '.format(\
                mainFileName,\
                addrFile,\
                vmtFile,\
                pstFile,\
                pitFile,\
                tfcFile,\
                topoFile,\
                chlFile,\
                workloadFile,\
                monitorFile,\
                parameterFile,\
                probeFile,\
                reorderFile,\
                loadFactor,\
                simStartInSec,\
                simEndInSec,\
                flowLaunchEndInSec,\
                pathSelStrategy,\
                pathExpiredTimeThldInNs,\
                probeStrategy,\
                piggyLatencyCnt,\
                piggybackStrategy,\
                flowletTimeoutInNs,\
                pathSelNum,\
                reTxThreshold,\
                probeTimeIntervalInNs,\
                args.srcNodeIdx,\
                args.dstNodeIdx,\
                args.sendingPktNum,\
                args.sendingIntervalInNs\
                )
        print(Line_command)
        os.system(Line_command)

os.chdir(vm_root_path + "outputFiles/")
cmd_compress = "tar -czvf" + "  " + "../compressedFiles/" + experimentalName + ".tar.gz" + "  ./" + experimentalName
print("Running Command:", cmd_compress)
os.system(cmd_compress)

os.chdir(vm_root_path + "analyzedCode/")
experimentalIdx = experimentalName.replace("A", "")
cmd_analyze = "python3 ./run_in_server.py --startIdx=" + experimentalIdx + " " \
              + "--endIdx=" + experimentalIdx+ " " \
              + "--srcDir=" + vm_root_path + "outputFiles/" + " " \
              + "--dstDir=" + vm_root_path + "analyzedResults/"
print("Running Command:", cmd_analyze)
os.system(cmd_analyze)





