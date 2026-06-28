#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
"""

import sys
import os
import subprocess
import time
import glob
import shutil
import argparse
import const_var
import ascendc_impl_build
import ascendc_bin_param_build
import ascendc_op_info


class CompileKernel:
    def __init__(self: any, args: any):
        self.op_type = args.op_name
        self.op_cpp_file = os.path.realpath(args.src_file)
        self.op_soc_ver = args.compute_unit
        self.compile_options = args.compile_options
        self.op_debug_config = args.debug_config
        self.op_cfg_ini = os.path.realpath(args.config_ini)
        self.op_tiling = os.path.realpath(args.tiling_lib)
        self.op_output = os.path.realpath(args.output_path)
        self.op_impl_py = None
        self.compile_sh = []
        self.build_opp_path = ""
        self.working_dir = os.path.join(
            os.getcwd(),
            self.op_type + "_" + self.op_soc_ver + "_" + str(time.time_ns()),
        )
        os.makedirs(self.working_dir)
        os.makedirs(self.op_output, exist_ok=True)

    def clean(self: any):
        shutil.rmtree(self.working_dir)
        return

    def ascendc_gen_impl(self: any):
        rep_cfg = {}
        rep_cfg[const_var.REPLAY_BATCH] = ""
        rep_cfg[const_var.REPLAY_ITERATE] = ""
        cfg_dir = {}
        cfg_dir[const_var.CFG_IMPL_DIR] = os.path.dirname(self.op_cpp_file)
        cfg_dir[const_var.CFG_OUT_DIR] = self.working_dir
        cfg_dir[const_var.AUTO_GEN_DIR] = os.path.dirname(self.op_cfg_ini)
        ascendc_impl_build.write_scripts(
            self.op_cfg_ini, rep_cfg, cfg_dir, [self.op_type], self.compile_options
        )
        py_files = glob.glob(os.path.join(self.working_dir, "dynamic", "*.py"))
        if py_files is None or len(py_files) != 1:
            self.clean()
            raise RuntimeError("compile py file {} generated error!".format(py_files))
        self.op_impl_py = os.path.join(
            self.working_dir, "dynamic", self.op_type + ".py"
        )
        os.rename(py_files[0], self.op_impl_py)
        if not os.path.exists(self.op_impl_py):
            self.clean()
            raise RuntimeError(
                "compile py file {} not generated!".format(self.op_impl_py)
            )

    def ascendc_gen_param(self: any):
        bin_param_path = os.path.join(self.working_dir, "bin_param")
        os.makedirs(bin_param_path)
        ascendc_bin_param_build.gen_bin_param_file(
            self.op_cfg_ini, bin_param_path, self.op_soc_ver, "", "", [self.op_type]
        )
        bin_param_files = glob.glob(os.path.join(bin_param_path, "*.json"))
        if bin_param_files is None or len(bin_param_files) <= 0:
            self.clean()
            raise RuntimeError("compile binary param json file not generated!")
        self.compile_sh = glob.glob(os.path.join(bin_param_path, "*.sh"))
        if self.compile_sh is None or len(self.compile_sh) != len(bin_param_files):
            self.clean()
            raise RuntimeError("compile binary shell file not generated!")

    def ascendc_put_tiling(self: any):
        self.build_opp_path = os.path.join(self.working_dir, "customize")
        tiling_path = os.path.join(
            self.build_opp_path, "op_impl", "ai_core", "tbe", "op_tiling"
        )
        os.makedirs(tiling_path)
        tiling_so = os.path.join(tiling_path, "liboptiling.so")
        os.symlink(self.op_tiling, tiling_so)
        if not os.path.exists(tiling_so):
            self.clean()
            raise RuntimeError("prepare tiling lib {} link failed!".format(tiling_so))

    def ascendc_build(self: any):
        op_info = ascendc_op_info.OpInfo(self.op_type, self.op_cfg_ini)
        op_file = op_info.get_op_file()
        op_bin_dir = os.path.join(self.op_output, self.op_soc_ver, op_file)
        os.makedirs(op_bin_dir, exist_ok=True)
        all_tar = []
        sub_cmd = []
        index = 0
        for sh in self.compile_sh:
            tar = op_file + str(index)
            build_path = os.path.join(self.working_dir, "kernel_" + str(index))
            os.makedirs(build_path)
            all_tar.append(tar)
            sub_cmd.append(tar + ":")
            sub_cmd.append(
                "\techo $(MAKE) && cd {} && bash {} $(PY) $(OUT) $(CPP)".format(
                    build_path, sh
                )
            )
            index += 1
        mkfile = os.path.join(self.working_dir, op_file + ".make")
        with os.fdopen(os.open(mkfile, const_var.WFLAGS, const_var.WMODES), "w") as fd:
            fd.write("export HI_PYTHON=python3\n")
            fd.write("export ASCEND_CUSTOM_OPP_PATH={}\n".format(self.build_opp_path))
            fd.write("export TILINGKEY_PAR_COMPILE=1\n")
            sub_cmd.insert(0, "all: " + " ".join(all_tar))
            fd.write("\n".join(sub_cmd))
        subprocess.run(['make', '-f', mkfile, 'PY=' + self.op_impl_py, 'OUT=' + op_bin_dir, 'CPP=' + self.op_cpp_file])


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--op-name", nargs="?", help="Op name(Camel string) to compile."
    )
    parser.add_argument("-s", "--src-file", nargs="?", help="Op kernel source file.")

    parser.add_argument("-u", "--compute-unit", nargs="?", help="Compute unit.")
    parser.add_argument(
        "-c", "--compile-options", nargs="?", help="Compile options of compiler."
    )
    parser.add_argument(
        "-d",
        "--debug-config",
        nargs="?",
        help="Debug config of op, ref opc op-debug-config.",
    )
    parser.add_argument("-i", "--config-ini", nargs="?", help="Op config ini file.")
    parser.add_argument(
        "-t", "--tiling-lib", nargs="?", help="Tiling shared library file."
    )

    parser.add_argument(
        "-o", "--output-path", nargs="?", help="Output path of compile result."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = args_parse()
    kernel_builder = CompileKernel(args)
    kernel_builder.ascendc_gen_impl()
    kernel_builder.ascendc_gen_param()
    kernel_builder.ascendc_put_tiling()
    kernel_builder.ascendc_build()
    kernel_builder.clean()
