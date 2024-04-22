## Triton Server Ensemble

This is the triton server set up with ensemle methods.

### List for model in model repo

1. Cell Counting Faster RCNN Model
2. Post Processing Python Wrapper
3. Ensemble Model with combined both cell counting and post processing models.

### To Run

```bash
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /data/model_repo:/model_repo nvcr.io/nvidia/tritonserver:24.01-py3 tritonserver --model-repository=/model_repo
```