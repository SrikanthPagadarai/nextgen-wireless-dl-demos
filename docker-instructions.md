# Build & Run

### Build
#### CPU build
`$ docker build -t sionna-dl-6g --build-arg POETRY_WITH=cpu .`
#### GPU build
`$ docker build -t sionna-dl-6g --build-arg POETRY_WITH=gpu .`

### Run on a CPU-only host
`$ docker run -it sionna-dl-6g bash`

### Run on a GPU host (with NVIDIA Container Toolkit installed)
`$ docker run -it --gpus all sionna-dl-6g bash`