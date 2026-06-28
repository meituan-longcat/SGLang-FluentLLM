#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
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


class ABI3Wheel(bdist_wheel):
    """
    Force to use abi3 tag for wheel, this wheel supports multiple python versions >= 3.8
    """
    def get_tag(self):
        python, abi, plat = super().get_tag()
        python = "cp38"
        abi = "abi3"
        return python, abi, plat

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

        # Get NPU_ARCH from environment variable or set default
        NPU_ARCH = os.environ.get('NPU_ARCH', 'dav-3510')
        logging.info(f"Using NPU_ARCH: {NPU_ARCH}")

        # Build the CMake project
        build_temp = os.path.join(os.getcwd(), 'build')
        cmake_config_command = ['cmake', '-S', os.getcwd(), '-B', build_temp,
                                '-DCMAKE_BUILD_TYPE=Release',
                                f'-DTorch_DIR={Torch_DIR}',
                                f'-DTORCH_NPU_PATH={TORCH_NPU_PATH}',
                                f'-DNPU_ARCH={NPU_ARCH}'
                                ]
        subprocess.check_call(cmake_config_command, cwd=os.getcwd())
        subprocess.check_call(['cmake', '--build', build_temp, '--config', 'Release', '--parallel', num_jobs], cwd=os.getcwd())
        logging.info("CMake extensions built successfully.")


cmdclass = {
    'clean': CleanCommand,
    'bdist_wheel': ABI3Wheel,
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
