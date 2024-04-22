# for GPU make sure -DTRITON_ENABLE_GPU=ON
mkdir build
cd build
cmake -DTRITON_ENABLE_GPU=ON -DTRITON_BACKEND_REPO_TAG=r24.01 -DTRITON_COMMON_REPO_TAG=r24.01 -DTRITON_CORE_REPO_TAG=r24.01 -DCMAKE_INSTALL_PREFIX:PATH=/data/model_repo/triton_post_process/install ..
make install