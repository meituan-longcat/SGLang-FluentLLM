#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created on Feb  28 20:56:45 2020
Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
"""

import argparse
import sys
import os
import json
import hashlib
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import const_var
import opdesc_parser
 
PYF_PATH = os.path.dirname(os.path.realpath(__file__))


class BinParamBuilder(opdesc_parser.OpDesc):
    def __init__(self: any, op_type: str):
        super().__init__(op_type)
        self.soc = ''
        self.out_path = ''
        self.tiling_keys = set()
        self.op_debug_config = ''

    def set_soc_version(self: any, soc: str):
        self.soc = soc

    def set_out_path(self: any, out_path: str):
        self.out_path = out_path

    def set_tiling_key(self: any, tiling_key_info: Set):
        if tiling_key_info:
            self.tiling_keys.update(tiling_key_info)

    def set_op_debug_config(self: any, op_debug_config: str):
        if op_debug_config:
            self.op_debug_config = op_debug_config

    def gen_input_json(self: any):
        key_map = {}
        count = len(self.input_dtype[0].split(','))
        required_parameters = set()
        index_value = -1

        for i in range(0, count):
            inputs = []
            outputs = []
            attrs = []
            required_parameter = []
            op_node = {}

            for idx in range(0, len(self.input_name)):
                idtypes = self.input_dtype[idx].split(',')
                ifmts = self.input_fmt[idx].split(',')
                itype = self.input_type[idx]
                para = {}
                para['name'] = self.input_name[idx][:-5]
                para['index'] = idx
                para['dtype'] = idtypes[i]
                para['format'] = ifmts[i]
                para['paramType'] = itype
                para['shape'] = [-2]
                para['format_match_mode'] = 'FormatAgnostic'

                if itype == 'dynamic':
                    inputs.append([para])
                    required_parameter.append(idtypes[i])
                elif itype == 'required':
                    inputs.append(para)
                    required_parameter.append(idtypes[i])
                else:
                    inputs.append(para)

            for idx in range(0, len(self.output_name)):
                odtypes = self.output_dtype[idx].split(',')
                ofmts = self.output_fmt[idx].split(',')
                otype = self.output_type[idx]
                para = {}
                para['name'] = self.output_name[idx][:-5]
                para['index'] = idx
                para['dtype'] = odtypes[i]
                para['format'] = ofmts[i]
                para['paramType'] = otype
                para['shape'] = [-2]
                para['format_match_mode'] = 'FormatAgnostic'
                if otype == 'dynamic':
                    outputs.append([para])
                    required_parameter.append(odtypes[i])
                elif otype == 'required':
                    outputs.append(para)
                    required_parameter.append(odtypes[i])
                else:
                    outputs.append(para)

            for attr in self.attr_list:
                att = {}
                att['name'] = attr
                atype = self.attr_val.get(attr).get('type').lower()
                att['dtype'] = atype
                att['value'] = const_var.ATTR_DEF_VAL.get(atype)
                attrs.append(att)

            required_parameter_tuple = tuple(required_parameter)
            if required_parameter_tuple in required_parameters:
                continue
            else:
                required_parameters.add(required_parameter_tuple)
                index_value +=1

            op_node['bin_filename'] = ''
            op_node['inputs'] = inputs
            op_node['outputs'] = outputs
            if len(attrs) > 0:
                op_node['attrs'] = attrs

            param = {}
            param['op_type'] = self.op_type
            param['op_list'] = [op_node]
            objstr = json.dumps(param, indent='  ')
            md5sum = hashlib.md5(objstr.encode('utf-8')).hexdigest()
            while key_map.get(md5sum) is not None:
                objstr += '1'
                md5sum = hashlib.md5(objstr.encode('utf-8')).hexdigest()
            key_map[md5sum] = md5sum
            bin_file = self.op_type + '_' + md5sum
            op_node['bin_filename'] = bin_file
            param_file = os.path.join(self.out_path, bin_file + '_param.json')
            param_file = os.path.realpath(param_file)
            with os.fdopen(os.open(param_file, const_var.WFLAGS, const_var.WMODES), 'w') as fd:
                json.dump(param, fd, indent='  ')
            self._write_buld_cmd(param_file, bin_file, index_value)


    def _write_buld_cmd(self: any, param_file: str, bin_file: str, index: int):
        hard_soc = const_var.SOC_MAP_EXT.get(self.soc)
        if not hard_soc:
            hard_soc = soc.capitalize()
        name_com = [self.op_type, self.op_file, str(index)]
        compile_file = os.path.join(self.out_path, '-'.join(name_com) + '.sh')
        compile_file = os.path.realpath(compile_file)

        bin_cmd_str = 'opc $1 --main_func={fun} --input_param={param} --soc_version={soc} \
                --output=$2 --impl_mode={impl} --simplified_key_mode=0 --op_mode=dynamic '

        build_cmd_var = "#!/bin/bash\n"
        build_cmd_var += f'echo "[{hard_soc}] Generating {bin_file} ..."\n'
        build_cmd_var += const_var.SRC_ENV
        build_cmd_var += bin_cmd_str.format(fun=self.op_intf, soc=hard_soc, param=param_file, 
                                           impl='high_performance,optional')
        if self.tiling_keys:
            tiling_keys_list = sorted(list(self.tiling_keys))
            tiling_key_str = ','.join([str(_key) for _key in tiling_keys_list])
            build_cmd_var += f' --tiling_key="{tiling_key_str}"'

        if self.op_debug_config:
            build_cmd_var += f' --op_debug_config={self.op_debug_config}'

        build_cmd_var += "\n"
        build_cmd_var += const_var.CHK_CMD.format(res_file=bin_file + '.json')
        build_cmd_var += const_var.CHK_CMD.format(res_file=bin_file + '.o')
        build_cmd_var += f'echo "[{hard_soc}] Generating {bin_file} Done"\n'

        with os.fdopen(os.open(compile_file, const_var.WFLAGS, const_var.WMODES), 'w') as fd:
            fd.write(build_cmd_var)


def get_tiling_keys(tiling_keys: str) -> Set:
    all_tiling_keys = set()
    if not tiling_keys:
        return all_tiling_keys

    tiling_key_list = tiling_keys.split(';')
    for tiling_key_value in tiling_key_list:
        pattern = r"(?<![^\s])(\d+)-(\d+)(?![^\s])"
        results = re.findall(pattern, tiling_key_value)
        if results:
            start, end = results[0]
            if int(start) > int(end):
                continue
            for i in range(int(start), int(end) + 1):
                all_tiling_keys.add(i)
        elif tiling_key_value.isdigit():
            all_tiling_keys.add(int(tiling_key_value))
    return all_tiling_keys


def parse_tiling_keys(tiling_key_file: str, soc: str) -> Dict:
    tiling_key_info = defaultdict(set)
    if not tiling_key_file:
        return tiling_key_info

    if not os.path.exists(tiling_key_file):
        return tiling_key_info

    with open(tiling_key_file, 'r') as file:
        contents = file.readlines()

    for _content in contents:
        content = _content.strip()
        op_tiling_key = content.split(',')
        if len(op_tiling_key) != 3:
            continue

        op_type = op_tiling_key[0]
        if not op_type:
            continue

        compute_unit = op_tiling_key[1]
        if compute_unit:
            compute_unit_list = compute_unit.split(';')
            if soc not in compute_unit_list:
                continue

        tiling_keys = op_tiling_key[2]
        format_tiling_keys = get_tiling_keys(tiling_keys)
        if format_tiling_keys:
            tiling_key_info[op_type].update(format_tiling_keys)

    return tiling_key_info


def gen_bin_param_file(cfgfile: str, out_dir: str, soc: str,
                        tiling_keys: str = '', op_debug_config: str = '', ops: list = None):
    if not os.path.exists(cfgfile):
        print(f'INFO: {cfgfile} does not exists in this project, skip generating compile commands.')
        return

    op_descs = opdesc_parser.get_op_desc(cfgfile, [], [], BinParamBuilder, ops)
    tiling_key_info = parse_tiling_keys(tiling_keys, soc)

    all_soc_key = "ALL"
    for op_desc in op_descs:
        op_desc.set_soc_version(soc)
        op_desc.set_out_path(out_dir)
        if op_debug_config:
            op_desc.set_op_debug_config(op_debug_config)
        if op_desc.op_type in tiling_key_info:
            op_desc.set_tiling_key(tiling_key_info[op_desc.op_type])
        if all_soc_key in tiling_key_info:
            op_desc.set_tiling_key(tiling_key_info[all_soc_key])
        op_desc.gen_input_json()


def parse_args(argv):
    """Command line parameter parsing"""
    parser = argparse.ArgumentParser()
    parser.add_argument('argv', nargs='+')
    parser.add_argument('--tiling-keys', nargs='?', const='', default='')
    parser.add_argument('--op-debug-config', nargs='?', const='', default='')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args(sys.argv)
    if len(args.argv) <= 3:
        raise RuntimeError('arguments must greater than 3')
    gen_bin_param_file(args.argv[1],
                    args.argv[2],
                    args.argv[3],
                    tiling_keys=args.tiling_keys,
                    op_debug_config=args.op_debug_config)
