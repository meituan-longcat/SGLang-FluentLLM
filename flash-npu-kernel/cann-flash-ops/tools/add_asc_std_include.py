#!/usr/bin/env python3
"""Add official CANN ASC headers to wrappers that explicitly require them.

Older opbuild generators only add tikcpp/tikcfw. CANN 9 moved the Std
algorithm headers under asc/include. Resolve this path from the *runtime*
compiler SDK, never embed the build machine's absolute SDK location.
"""
import ast
import sys
from pathlib import Path

ANCHOR = '    options.append("-I" + os.path.join(tikcpp_path, "..", "ascendc", "act"))\n'
EXTRA = (
    '    # Official CANN 9 Std headers; keep legacy tikcpp headers first.\n'
    '    for sdk_root in (os.path.dirname(tikcpp_path), os.path.dirname(os.path.dirname(tikcpp_path))):\n'
    '        asc_std_include = os.path.join(sdk_root, "asc", "include")\n'
    '        if os.path.isfile(os.path.join(asc_std_include, "utils", "std", "algorithm.h")):\n'
    '            options.append("-I" + asc_std_include)\n'
    '            break\n'
)

def patch(dynamic_dir):
    # Only operators explicitly requiring CANN9 ASC Std/Reduce headers. Keep
    # the validated SFA1 operators' compiler search paths completely unchanged.
    for name in ("mlp_dflash_infer_attn", "moe_distribute_combine_v2", "moe_distribute_dispatch_v2",
                 "moe_gating_top_k_expert_map"):
        target = Path(dynamic_dir) / (name + '.py')
        if not target.is_file():
            continue
        extra = EXTRA
        if name.startswith("moe_"):
            extra = extra.replace('            options.append("-I" + asc_std_include)\n',
                '            options.append("-I" + asc_std_include)\n'
                '            options.append("-I" + os.path.join(sdk_root, "asc"))\n')
        source = target.read_text()
        if extra in source:
            continue
        if source.count(ANCHOR) != 1:
            raise RuntimeError('Unsupported opbuild wrapper ASC include anchor: ' + name)
        source = source.replace(ANCHOR, ANCHOR + extra)
        ast.parse(source)
        target.write_text(source)

if __name__ == '__main__':
    patch(sys.argv[1])
