#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import subprocess
import logging
from setuptools import setup, find_packages, Distribution, Command
from wheel.bdist_wheel import bdist_wheel

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
PACKAGE_NAME = "flash_npu_kernel"
VERSION = "1.0.0"
DESCRIPTION = "PyTorch Ascend C operator extensions"


class CleanCommand(Command):
    """
    usage: python setup.py clean
    """
    description = "Clean build artifacts from the source tree"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        folders_to_remove = ['build', 'dist', f'{PACKAGE_NAME}.egg-info']
        for folder in folders_to_remove:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                logging.info(f"Removed folder: {folder}")
        for root, _, files in os.walk('.'):
            for file in files:
                if file.endswith(('.pyc', '.pyo')):
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    logging.info(f"Removed file: {file_path}")
        logging.info("Cleaned build artifacts.")


class BinaryDistribution(Distribution):
    """
    Make this wheel not a pure python package
    """
    def is_pure(self):
        return False

    def has_ext_modules(self):
        return True


class BinaryWheel(bdist_wheel):
    def get_tag(self):
        _, _, plat = super().get_tag()
        cpver = f"cp{sys.version_info.major}{sys.version_info.minor}"
        return cpver, cpver, plat

    def run(self):
        self.run_command('cmake_build')
        super().run()


class CMakeBuildCommand(Command):
    """
    Custom command to build CMake extensions
    """
    description = "Build CMake extensions"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        cpu_count = os.cpu_count() or 2
        num_jobs = str(cpu_count)
        # Get Torch and Torch NPU paths
        import torch
        TORCH_CMAKE_PATH = torch.utils.cmake_prefix_path
        Torch_DIR = os.path.join(TORCH_CMAKE_PATH, "Torch")
        logging.info(f"Using Torch path: {Torch_DIR}")
        import torch_npu
        TORCH_NPU_PATH = os.path.dirname(torch_npu.__file__)
        logging.info(f"Using Torch NPU path: {TORCH_NPU_PATH}")

        # Get NPU_ARCH from environment variable or auto-detect from hardware
        NPU_ARCH = os.environ.get('NPU_ARCH')
        if not NPU_ARCH:
            _soc_to_arch = {
                'Ascend910B': 'dav-2201',
                'Ascend910_93': 'dav-2201',
                'Ascend950': 'dav-3510',
            }
            try:
                if torch.npu.is_available():
                    device_name = torch.npu.get_device_name(0)
                    for soc_prefix, arch_name in _soc_to_arch.items():
                        if device_name.startswith(soc_prefix):
                            NPU_ARCH = arch_name
                            break
                if not NPU_ARCH:
                    NPU_ARCH = 'dav-2201'
            except Exception:
                NPU_ARCH = 'dav-2201'
            logging.info(f"NPU_ARCH not set, auto-detected: {NPU_ARCH}")
        NPU_ARCH_CMAKE = NPU_ARCH.replace(',', ';')
        logging.info(f"Using NPU_ARCH: {NPU_ARCH} -> CMake: {NPU_ARCH_CMAKE}")

        build_temp = os.path.join(os.getcwd(), 'build')
        cmake_config_command = ['cmake', '-S', os.getcwd(), '-B', build_temp,
                                '-DCMAKE_BUILD_TYPE=Release',
                                f'-DTorch_DIR={Torch_DIR}',
                                f'-DTORCH_NPU_PATH={TORCH_NPU_PATH}',
                                f'-DNPU_ARCH={NPU_ARCH_CMAKE}'
                                ]
        subprocess.check_call(cmake_config_command, cwd=os.getcwd())
        subprocess.check_call(['cmake', '--build', build_temp, '--config', 'Release', '--parallel', num_jobs], cwd=os.getcwd())
        arch_list = [a.replace('-', '_') for a in NPU_ARCH.split(',')]
        pkg_dir = os.path.join(os.getcwd(), PACKAGE_NAME)
        for f in os.listdir(pkg_dir):
            if f.endswith('.so') and not any(f'_{a}.so' in f for a in arch_list):
                os.remove(os.path.join(pkg_dir, f))
                logging.info(f"Removed stale .so: {f}")
        logging.info("CMake extensions built successfully.")


cmdclass = {
    'clean': CleanCommand,
    'bdist_wheel': BinaryWheel,
    'cmake_build': CMakeBuildCommand,
}


setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    packages=find_packages(),
    package_data={PACKAGE_NAME: ['*.so']},
    distclass=BinaryDistribution,
    cmdclass=cmdclass,
    zip_safe=False,
    install_requires=[
        "torch",
        "torch_npu"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires='>=3.8',
)
