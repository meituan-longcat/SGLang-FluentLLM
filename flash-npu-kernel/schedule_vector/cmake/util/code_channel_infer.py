#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created on Feb  28 20:56:45 2020
Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
"""
import os
import stat
import ctypes
import collections
import shutil
import subprocess
import copy

"""CODE_* is used to cube/vector api is called in operator code
CODE_MIX means both cube and vector api is called
CODE_CUBE means only cube api is called
CODE_VEC means only vector api is called
"""
CODE_MIX = 0
CODE_CUBE = 1
CODE_VEC = 2


def _is_v220(op_product: str):
    """return if current soc version is V220

    Returns:
        res: True means V220
    """
    if op_product in ["ascend910b", "ascend910_93"]:
        return True
    return False


InfoCodeChanelParams = collections.namedtuple('InfoCodeChanelParams',\
['src_file', 'tiling_header', 'kernel_name', 'outdir', 'op_product', 'compile_options'])


def infer_code_channel(params: InfoCodeChanelParams):
    """get code channel for v220, return CODE_MIX if soc version is not V220

    Args:
        src_file (str): AscendC operator code file
        src_file (str): AscendC operator tiling header file
        kernel_name (str): kernel function name
        optype (str): operator type
        compile_options (list): compile options for bisheng cmd

    Raises:
        Exception: if not exist L1/L0/UB if code, it's not a aicore code

    Returns:
        res (int): CODE_MIX/CODE_CUBE/CODE_VEC
    """
    if not _is_v220(params.op_product):
        return CODE_MIX
    return CODE_VEC
    if params.compile_options is None:
        compile_options = []
    else:
        compile_options = params.compile_options
    bisheng = shutil.which("bisheng")
    if bisheng is not None:
        bisheng_path = os.path.dirname(bisheng)
        tikcpp_path = os.path.realpath(os.path.join(bisheng_path, "..", "..", "tikcpp"))
    else:
        tikcpp_path = os.path.realpath("/usr/local/Ascend/latest/compiler/tikcpp")
    compile_options.append("-I" + tikcpp_path)
    compile_options.append("-I" + os.path.join(tikcpp_path, "tikcfw"))
    compile_options.append("-I" + os.path.join(tikcpp_path, "tikcfw", "impl"))
    compile_options.append("-I" + os.path.join(tikcpp_path, "tikcfw", "interface"))
    compile_options += ["-include", params.tiling_header]
    arch = "dav-c220-cube"
    sub_core_type = "AIC"
    optional_core = "AiCore"
    compile_cmd = [shutil.which("bisheng"), '-c', '-O3']
    compile_cmd += compile_options
    temp_file_name_tag = "_" + str(os.getpid()) + "_temp.o"
    dst_file = os.path.join(kernel_meta_dir, kernel_name + temp_file_name_tag)
    compile_cmd += [params.src_file, "--cce-aicore-arch={}".format(arch),
                    "--cce-aicore-only", "-o", dst_file,
                    "-mllvm", "-cce-aicore-function-stack-size=16000",
                    "-mllvm", "-cce-aicore-record-overflow=true",
                    "-mllvm", "-cce-aicore-addr-transform"]
    compile_cmd += ["-std=c++17"]
    print('get_code_channel: ', ' '.join(compile_cmd))
    proc = subprocess.Popen(
        compile_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    if proc.returncode != 0:
        print('get_code_channel coretype compile error: ', out.decode())
        msg = "compile %s error :%s\n" % (params.src_file, out.decode())
        raise Exception(f"get_code_channel coretype error, msg is{msg}")
    objdump_cmd = ['objdump', '-s', '-j', '.text', '{}'.format(dst_file)]

    proc = subprocess.Popen(
        objdump_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    if proc.returncode != 0:
        print('get_code_channel objdump error: ', out.decode())
        msg = "get_code_channel objdump %s error :%s\n" % (src_file, out.decode())
        raise Exception(f"get_code_channel objdump error, msg is{msg}")
    os.remove(dst_file)
    lines = out.decode('utf-8').split('\n')
    for line in lines:
        insts = line.strip().split()
        if len(insts) < 5:
            continue
        for inst in insts[1:]:
            if len(inst) != 8:
                continue
            if inst[6] == 'f':
                return CODE_MIX
    return CODE_VEC
