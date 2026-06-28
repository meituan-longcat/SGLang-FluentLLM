pip3 install \
	openai_harmony==0.0.8 \
	pybase64==1.4.3 \
	numba==0.65.1 \
	ray==2.55.1 \
	attrs==26.1.0 \
	cython==3.2.6 \
	numpy==1.26.4 \
	decorator==5.3.1 \
	sympy==1.13.1 \
	cffi==2.0.0 \
	pyyaml==6.0.3 \
	pathlib2==2.3.7.post1 \
	psutil==7.2.2 \
	protobuf==3.20.3 \
	scipy==1.17.1 \
	requests==2.34.2 \
	absl-py==2.4.0 \
	wheel==0.47.0 \
	setuptools==65.5.0 \
	cloudpickle==3.1.1 \
	ml-dtypes==0.5.1 \
	tornado==6.5.7 \
	rich==14.0.0 \
	easydict==1.13 

pip3 install torch==2.6.0+cpu \ 
             torchvision==0.21.0 \
             --index-url https://download.pytorch.org/whl/cpu \
             --trusted-host download.pytorch.org
 
pip3 install \
    pulp==2.7 \
    pystack==1.4.1 \
    nltk==3.8.1 \
    pybind11==2.11.1 \
    einops==0.7.0 \
    tensorboard==2.19.0 \
    sentencepiece==0.2.0 \
    flask==3.0.0 \
    flask-restful==0.3.10 \
    transformers==4.51.3 \
    openai==1.47.1 \
    msgspec==0.18.6 \
    gguf==0.10.0 \
    mistral-common==1.4.3 \
    annotated_types==0.7.0 \
    pydantic_core==2.46.4 \
    prometheus_client==0.22.1 \
    py-cpuinfo==9.0.0 \
    dynamo==0.1.1 \
    jsonlines==3.1.0 \
    jieba==0.42.1 \
    langdetect==1.0.9 \
    antlr4-python3-runtime==4.13.1 \
    word2number==1.1 \
    timeout-decorator==0.5.0 

pip3 install pydantic==2.13.4 --no-deps 

pip3 install numpy==1.26.4 \
    torchvision==0.21.0 \
    torchao==0.9.0 \
    torchaudio==2.6.0 \
    transformers==4.51.0 \
    aiolimiter==1.2.1 \
    rich==14.0.0 \
    triton==3.0.0 \
    pyzmq==26.4.0 \
    ipython==9.15.0 \
    orjson==3.11.9 \
    setproctitle==1.3.7 \
    attrs==26.1.0 \
    cloudpickle==3.1.1 \
    decorator==5.3.1 \
    psutil==7.2.2 \
    scipy==1.17.1 \
    synr==0.5.0 \
    pyyaml==6.0.3 \
    wheel==0.47.0 \
    setuptools==65.5.0 \
    setuptools-scm==10.2.0 \
    cmake==4.3.4 \
    ninja==1.13.0 \
    blake3==1.0.5 \
    aiohttp==3.12.13 \
    opencv-python-headless==4.11.0.86 \
    openai==1.86.0 \
    compressed-tensors==0.9.0 \
    fastapi==0.115.12 \
    uvicorn==0.34.2 \
    uvloop==0.21.0 \
    dill==0.4.0 \
    outlines==0.1.11 \
    partial-json-parser==0.2.1.1.post5 \
    python-multipart==0.0.20 

#yum -y install numactl numactl-devel && \
pip3 install --no-deps vllm==0.7.1 

pip3 install mooncake_transfer_engine==0.3.9 --force-reinstall --no-deps

#yum install -y python3-devel 
pip3 install \
    py-spy==0.4.2 \
    xgrammar==0.1.27 \
    pybase64==1.4.3 \
    datasets==5.0.0 
    # python-cat==0.0.12 \
