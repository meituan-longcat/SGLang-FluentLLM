: "${FLUENTLLM_HOME:?Set FLUENTLLM_HOME to the FluentLLM checkout directory before running this script.}"
export FLUENTLLM_HOME
export EPS_HOME=${FLUENTLLM_HOME}/3rdparty/eps/
export PYTHONPATH=${EPS_HOME}/python/:$PYTHONPATH
export PYTHONPATH=${FLUENTLLM_HOME}/python:${PYTHONPATH}
export PYTHONPATH=${FLUENTLLM_HOME}/:${PYTHONPATH}
export LD_PRELOAD=/usr/lib64/libcuda.so
export LD_LIBRARY_PATH=/usr/local/lib64:/usr/local/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvshmem/lib:$LD_LIBRARY_PATH
export SGLANG_MOE_CONFIG_DIR=${FLUENTLLM_HOME}/3rdparty/triton_configs/

python3 bench.py
