## Triton Server Ensemble

This is the triton server set up with ensemle methods.

### List for model in model repo

1. Cell Counting Faster RCNN Model
2. Post Processing Python Wrapper
3. Ensemble Model with combined both cell counting and post processing models.

### To Build for custom Python Backend
> Note: Triton has pre-build for python version 3.10. So, no need to rebuild the python backend for triton server if you are using python 3.10. If you are using different version of python then you need to build the python backend for triton server.
```bash	
git clone https://github.com/triton-inference-server/python_backend -b r<xx.yy>	# currently we are using r24.01
cd python_backend
# for GPU make sure -DTRITON_ENABLE_GPU=ON
# for CPU make sure -DTRITON_ENABLE_GPU=OFF
mkdir build
cd build
cmake -DTRITON_ENABLE_GPU=ON -DTRITON_BACKEND_REPO_TAG=r24.01 -DTRITON_COMMON_REPO_TAG=r24.01 -DTRITON_CORE_REPO_TAG=r24.01 -DCMAKE_INSTALL_PREFIX:PATH=/data/model_repo/triton_post_process/install ..
make install
```	

### To create tar file of the custom env for triton server

```bash
conda create -n tritonserver python=3.10 -y
conda activate tritonserver
pip install tensorflow 
conda install -c conda-forge libstdcxx-ng=12 -y
conda-pack
```

### To Run

```bash
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /data/model_repo:/model_repo nvcr.io/nvidia/tritonserver:24.01-py3 tritonserver --model-repository=/model_repo
```