"""Compile the real Host tiling headers; no Torch, CANN or device is needed.

Only the platform query and TORCH_CHECK are replaced with test doubles. The
two production calc_tiling_data functions run unchanged. Allocation checks
model the kernel's metadata slot and two independently aligned queue slots.
"""

import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLATFORM = r"""
#pragma once
#include <cstdint>
namespace platform_ascendc {
enum class CoreMemType { UB };
class PlatformAscendCManager {
public:
    inline static uint64_t ubSize = 196352;
    inline static int64_t cores = 64;
    static PlatformAscendCManager* GetInstance() {
        static PlatformAscendCManager platform;
        return &platform;
    }
    void GetCoreMemSize(CoreMemType, uint64_t& size) { size = ubSize; }
    int64_t GetCoreNumAiv() { return cores; }
};
}
"""
TORCH = r"""
#pragma once
#include <stdexcept>
#define TORCH_CHECK(condition, ...) \
    do { if (!(condition)) throw std::runtime_error("TORCH_CHECK"); } while (0)
"""
PROBE = r"""
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include "get_out_cache_loc/get_out_cache_loc_tiling.h"
#include "rearrange_accept_index/rearrange_accept_index_tiling.h"

static int64_t aligned(int64_t bytes) { return (bytes + 31) / 32 * 32; }
static void require(bool ok, const char* message) {
    if (!ok) throw std::logic_error(message);
}

template <typename Calculate>
void verify(Calculate calculate, int64_t budget, int64_t cores) {
    int64_t checks = 0;
    for (int64_t batch = 1; batch <= budget / 8 + 8; ++batch) {
        for (int64_t elem : {4, 8}) {
            // The existing Host contract conservatively reserves int64
            // metadata even for an int32 kernel. Keep that reservation.
            const int64_t reserved = aligned(batch * 8);
            bool rejected = false;
            try {
                const auto t = calculate(batch, 131070, elem);
                const int64_t tile = int64_t(t.ubFactor) * elem;
                require(tile > 0, "zero tile was accepted");
                require(tile % 32 == 0, "tile is not block aligned");
                require(reserved + 2 * tile <= budget, "reserved UB budget exceeded");
                require(aligned(batch * elem) + 2 * aligned(tile) <= budget,
                        "kernel InitBuffer allocation exceeds UB");
                require(reserved + 2 * (tile + 32) > budget,
                        "a full aligned double-buffer block was unnecessarily lost");
                require(t.batchSize == batch && t.poolLen == 131070,
                        "input metadata changed");
                require(t.usedCoreNum > 0 && t.usedCoreNum <= cores,
                        "invalid core partition");
                require(int64_t(t.usedCoreNum - 1) * t.blockFactor +
                            t.tailBlockFactor == batch,
                        "core partition does not cover the batch");
                if (budget == 196352 && batch == 1) {
                    require(tile == 98144, "BS1 regression: expected 98144 bytes per slot");
                }
            } catch (const std::runtime_error&) {
                rejected = true;
            }
            require(rejected == (reserved + 64 > budget),
                    "capacity rejection differs from minimum two-slot allocation");
            ++checks;
        }
    }
    std::cout << "PASS " << checks << " tilings\n";
}

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    const auto budget = std::stoll(argv[2]);
    const auto cores = std::stoll(argv[3]);
    platform_ascendc::PlatformAscendCManager::ubSize = budget;
    platform_ascendc::PlatformAscendCManager::cores = cores;
    try {
        if (std::string(argv[1]) == "get") {
            verify(flash::GetOutCacheLoc::calc_tiling_data, budget, cores);
        } else {
            verify(flash::RearrangeAcceptIndex::calc_tiling_data, budget, cores);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
"""


class TestUbTiling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        if not compiler or not shutil.which(compiler[0]):
            raise unittest.SkipTest("A C++17 compiler is required")
        cls.directory = tempfile.TemporaryDirectory(prefix="flash-ub-tiling-")
        cls.addClassCleanup(cls.directory.cleanup)
        root = Path(cls.directory.name)
        for name, source in {
            "platform/platform_ascendc.h": PLATFORM,
            "torch/all.h": TORCH,
            "probe.cpp": PROBE,
        }.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        cls.binary = root / "probe"
        subprocess.run(
            compiler
            + [
                "-std=c++17",
                "-O2",
                "-I" + str(root),
                "-I" + str(ROOT / "csrc"),
                str(root / "probe.cpp"),
                "-o",
                str(cls.binary),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def check_budget(self, budget, cores):
        for op in ("get", "rearrange"):
            with self.subTest(operator=op, budget=budget, cores=cores):
                result = subprocess.run(
                    [str(self.binary), op, str(budget), str(cores)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_910b_budget_and_core_partitions(self):
        for cores in (1, 8, 32, 64):
            self.check_budget(196352, cores)

    def test_capacity_boundaries(self):
        for budget in (32, 63, 64, 95, 96, 127, 128, 196351, 262144):
            self.check_budget(budget, 64)


if __name__ == "__main__":
    unittest.main()
