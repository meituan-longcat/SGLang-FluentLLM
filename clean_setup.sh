set -e

SM=${1:-sm90}

echo "${SM}"

if [ "${SM}" = "sm90" ]; then
    export FLASHINFER_CUDA_ARCH_LIST="9.0a"
elif [ "${SM}" = "sm80" ]; then
    export FLASHINFER_CUDA_ARCH_LIST="8.0"
fi

export MAX_JOBS=16
echo "install eps"
cd 3rdparty/eps
pip3 install -v . --no-build-isolation
cd -

pip3 install build nvidia-nvshmem-cu12==3.5.21 apache-tvm-ffi==0.1.5 nvidia-cudnn-frontend==1.16.0 nvidia-cutlass-dsl==4.3.2 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

echo "install flashinfer"
pip3 uninstall -y flashinfer-python flashinfer-jit-cache

cd 3rdparty/flashinfer
git submodule update --init --recursive
rm -rf $HOME/.cache/flashinfer/
rm -rf build
rm -rf LICENSE.*.txt
pip install -v .

cd flashinfer-jit-cache
FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST} pip install -v . --no-build-isolation
cd ../../..

if [ ${SM} = "sm90" ]; then
    echo "install flash_mla"
    pip3 uninstall -y flash_mla
    cd 3rdparty/flashmla
    git submodule update --init
    FLASH_MLA_DISABLE_SM100=1 pip3 install --no-build-isolation -v .
    cd -

    echo "install flash_mla_swap"
    pip3 uninstall -y flash_mla_swap
    cd 3rdparty/flashmla-swap
    git submodule update --init
    FLASH_MLA_DISABLE_FP16=1 pip install --no-build-isolation -v .
    cd -

    echo "install flash_mla_fp8"
    pip3 uninstall -y flash_mla_fp8
    cd 3rdparty/flashmla-fp8
    git submodule update --init
    FLASH_MLA_DISABLE_FP16=1 pip install --no-build-isolation -v .
    cd -

    echo "install deep_gemm"
    pip3 uninstall -y deep_gemm
    cd 3rdparty/deep_gemm
    git submodule update --init
    python3 setup.py install
    cd -

    echo "install deep_gemm_oss"
    pip3 uninstall -y deep_gemm_oss
    cd 3rdparty/deep_gemm_oss
    git submodule update --init --recursive
    sh install.sh
    cd -

    echo "install deep_ep"
    pip3 uninstall -y deep_ep
    cd 3rdparty/deep_ep
    NVSHMEM_DIR=/opt/nvshmem python3 setup.py install
    rm -rf deep_ep.egg-info/
    git checkout .
    cd -

    echo "install fast-hadamard-transform"
    pip3 uninstall -y fast_hadamard_transform
    cd 3rdparty/fast-hadamard-transform
    rm -rf fast_hadamard_transform.egg-info/ build/
    python3 setup.py install
    cd -

    echo "install fa3"
    pip3 uninstall -y flash-attn-3
    cd 3rdparty/flash-attention
    git submodule update --init
    cd hopper
    python3 setup.py install
    cd -
fi

pip3 install openai-harmony==0.0.3 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
unset MAX_JOBS
