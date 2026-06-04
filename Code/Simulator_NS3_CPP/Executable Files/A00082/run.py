import os
import sys
import argparse
import shutil


import os
import sys
import argparse
import shutil
import time

def copy_newer_files(src_dir, dst_dir):
    # 确保目标目录存在
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # 使用os.walk遍历源目录及其子目录
    for root, dirs, files in os.walk(src_dir): #root：当前目录的路径（字符串形式）。dirs：当前目录下子目录的名称列表（字符串列表）。files：当前目录下文件的名称列表（字符串列表）。
        # 计算相对于源目录的相对路径
        rel_root = os.path.relpath(root, src_dir)
        dst_subdir = os.path.join(dst_dir, rel_root)
        # 确保目标目录的子目录存在
        if not os.path.exists(dst_subdir):
            os.makedirs(dst_subdir)
        # 遍历文件
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_subdir, file)
            # 如果目标文件不存在或源文件更新时间晚于目标文件，则复制源文件到目标目录
            if not os.path.exists(dst_file) or os.path.getmtime(src_file) > os.path.getmtime(dst_file):
                shutil.copy2(src_file, dst_subdir)
                print(f"File {src_file} is newer or does not exist in destination, copied to {dst_subdir}.")

def override_files(src_dir, dst_dir):
    # 确保目标目录存在
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # 使用os.walk遍历源目录及其子目录
    for root, dirs, files in os.walk(src_dir): #root：当前目录的路径（字符串形式）。dirs：当前目录下子目录的名称列表（字符串列表）。files：当前目录下文件的名称列表（字符串列表）。
        # 计算相对于源目录的相对路径
        rel_root = os.path.relpath(root, src_dir)
        dst_subdir = os.path.join(dst_dir, rel_root)
        # 确保目标目录的子目录存在
        if not os.path.exists(dst_subdir):
            os.makedirs(dst_subdir)
        # 遍历文件
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_subdir, file)
            # 如果目标文件不存在或源文件更新时间晚于目标文件，则复制源文件到目标目录
            if os.path.exists(dst_file):
                os.remove(dst_file)   #递归删除文件夹下的所有子文件夹和子文件和其本身
                print(f"File {dst_file} is deleted.")

            shutil.copy2(src_file, dst_subdir)
            print(f"File {src_file} is copied to {dst_subdir}.")

vm_root_path = "/file-in-ctr/" if os.path.exists("/file-in-ctr/") else "/file-in-cntr/"
ns3_root_path = "/app/ns3-detnet-rdma-main/ns-3.33/"

# update the module files
sydDirList = ['src']
for sydDir in sydDirList:
    override_files(vm_root_path+sydDir, ns3_root_path+sydDir)



current_file_path = os.path.abspath(__file__)
current_dir_path = os.path.dirname(current_file_path)
current_dir_name = os.path.basename(current_dir_path)
experimentalName = current_dir_name
mainFileName   =  "main"

parser = argparse.ArgumentParser(description="请输入以下参数")
parser.add_argument("--configFileName", default="CONFIG_DCQCN.txt", help="defaultFileName, by default CONFIG.txt")
parser.add_argument("--topoFileName", default="TOPO.txt", help="defaultFileName, by default CONFIG.txt")
parser.add_argument("--simStartTimeInSec", default="0", help="defaultFileName, by default CONFIG.txt")
parser.add_argument("--simEndTimeInSec", default="5000", help="defaultFileName, by default CONFIG.txt")
parser.add_argument("--kvCacheFileName", default="GPT_1.txt", help="kvCacheFileName, by default OB_KV_INCAST.txt")

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

# # update the userdefinedfunction/model/
# if os.path.exists(ns3_userdefinedfunction_model_path):
#     shutil.rmtree(ns3_userdefinedfunction_model_path)   #递归删除文件夹下的所有子文件夹和子文件和其本身
# shutil.copytree(vm_userdefinedfunction_model_path, ns3_userdefinedfunction_model_path)


# create the sub dir for the results in outputFiles/
if not os.path.exists(vm_outputFiles_path):
    # shutil.rmtree(vm_outputFiles_path)   #递归删除文件夹下的所有子文件夹和子文件
    os.makedirs(vm_outputFiles_path)
# file_path = os.path.join(dir_path, 'A.txt')  # 文件路径





os.chdir(ns3_waf_path)
qlenMonitorIntervalInNs = 100000

for kvCacheFileName in ['GPT_1.txt', 'GPT_4.txt', 'GPT_8.txt',
                        'GPT_16.txt', 'GPT_32.txt',
                        'LLAM_1.txt', 'LLAM_4.txt', 'LLAM_8.txt',
                        'LLAM_16.txt', 'LLAM_32.txt']:
    
# for kvCacheFileName in ['GPT_1.txt', 'GPT_4.txt','GPT_8.txt',
                        # 'GPT_16.txt', 'GPT_32.txt']:

# for kvCacheFileName in ['LLAM_1.txt', 'LLAM_4.txt', 'LLAM_8.txt',
#                         'LLAM_16.txt', 'LLAM_32.txt']:

# for kvCacheFileName in ['LLAM_32.txt']:
    Line_command = '\
            ./waf --run "scratch/{}\
            --fileIdx={}\
            --outputFileDir={}\
            --inputFileDir={}\
            --topoFileName={}\
            --configFileName={}\
            --simStartTimeInSec={}\
            --kvCacheFileName={}\
            --qlenMonitorIntervalInNs={}\
            --simEndTimeInSec={}\
            "\
            '.format(\
            mainFileName,\
            experimentalName+"_"+kvCacheFileName[:-4],\
            vm_outputFiles_path,\
            vm_inputFiles_path,\
            vm_inputFiles_path + args.topoFileName,\
            vm_inputFiles_path + args.configFileName,\
            args.simStartTimeInSec,\
            vm_inputFiles_path + kvCacheFileName,\
            qlenMonitorIntervalInNs,\
            args.simEndTimeInSec\
                )
    print(Line_command)
    os.system(Line_command)





