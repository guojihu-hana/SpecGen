# The artifact evaluation's environment: CUDA toolkit (nvcc), PyTorch, Nsight Compute (ncu), SpecGen.
#
#   docker build -f ae/Dockerfile -t specgen-ae .
#   docker run --rm --gpus all --cap-add SYS_ADMIN --ipc host -e DEEPSEEK_API_KEY \
#       -v "$PWD/ae_results:/specgen/ae_results" specgen-ae getting-started   # = bash ae/run.sh getting-started
#   docker run ... specgen-ae e1                                              # = bash ae/run.sh e1
#
# --cap-add SYS_ADMIN lets ncu read the GPU performance counters.  Behind a proxy, add -e https_proxy.
FROM nvcr.io/nvidia/pytorch:26.03-py3
WORKDIR /specgen
COPY . .
RUN pip install --no-cache-dir -e ".[tokens,dev]"
ENTRYPOINT ["bash", "ae/run.sh"]
