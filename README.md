# AN5D-Artifact

This repository provides the artifact to manifest our paper "AN5D: Automated Stencil Framework for High-Degree Temporal Blocking on GPUs" (to be presented at CGO 2020). We evaluated the generated code by our framework AN5D (which is available at https://github.com/khaki3/AN5D). This artifact repository (https://github.com/khaki3/AN5D-Artifact) contains the original code converted to CUDA code and all "Tuned" code which are evaluated for our paper.

The artifacts for "Loop Tiling", "Hybrid Tiling" and "STENCILGEN" code are separately presented at https://github.com/khaki3/StencilBench/tree/const and https://github.com/khaki3/IEEE2017.

## Install with nvidia-docker
We provide Dockerfile to build a container of nvidia-docker. You would find three repositories inside the container for the artifact evaluation.
```
% docker build -t an5d/env .
% docker run -it --rm --pid=host --gpus all an5d/env
% ls | egrep 'AN5D-Artifact|StencilBench|IEEE2017'
```

## Install

At first, clone AN5D to your local machine:

```
% git clone 'https://github.com/khaki3/AN5D' && cd AN5D
```

Then, build it by following commands:

```
% ./get_submodules.sh && ./autogen.sh

% CC=gcc-4.8 CXX=g++-4.8 ./configure && make && make install
```

## Evaluation

First, clone our artifact repository:

```
% git clone 'https://github.com/khaki3/AN5D-Artifact' && cd AN5D-Artifact
```

To execute benchmarks with Sconf configuration:

```
% ./run_sconf.sh
```

To execute benchmarks with Tuned configuration:

```
% cd compiled

% export GPUNAME=v100

% export REGNUM=0

% ./nvcc_compile_top5.sh $(nproc) $REGNUM

% ./run.sh $GPUNAME

% cat $GPUNAME/float_run.csv 

% cat $GPUNAME/double_run.csv 
```

(The last two commands show the results. The variable REGNUM can be chosen from 0(unrestricted)/32/64/96 to limit register use per thread. The variable GPUNAME must be v100 or p100.)

### Evaluate all generated configurations on A100/H100

The generated CUDA sources under `compiled/float` and `compiled/double`
contain 146 configurations for each precision. They are the configurations
preserved in this artifact, not the complete parameter search space. The
following script compiles and runs every preserved configuration for register
limits 0 (unrestricted), 32, 64, and 96:

```
% ./benchmark_generated.sh --gpu a100
% ./benchmark_generated.sh --gpu h100
```

Pass one or more kernel names to test only those kernels:

```
% ./benchmark_generated.sh --gpu a100 j2d5pt star3d1r
```

A100 is compiled for `sm_80` and H100 for `sm_90`. Compilation concurrency can
be controlled with `--jobs N`. On a machine without a GPU, use
`--compile-only` to validate compilation without running a benchmark:

```
% ./benchmark_generated.sh --gpu h100 --compile-only j2d5pt
```

Results are written below `results/<gpu>/`. `all_results.csv` contains every
configuration, including failures, while `best_results.csv` contains the
highest measured GFLOPS for each kernel and precision. Runs use the Evaluation
workload (`-s 16384` for 2D, `-s 512` for 3D, and `-t 1000 -n 5`).


## Parameter Customizing

AN5D can be used as follows:

```
% an5d --bt=4 --bs1=64 --bs2=16 --sl=256 star3d1r.c

% nvcc --use_fast_math -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -Xptxas -v -Xcompiler -fopenmp -O3 ./star3d1r_*.cu

% ./a.out -t 8 -s 256 -n 2 -c
```

AN5D supports the following options:

```
Usage: an5d [OPTION...] input

      --bt=<size>             Temporal blocking size [default: 4]
      --bs1=<size>            Spatial blocking size along 1st dimension
                              (the fastest varying dimension) [default: 32]
      --bs2=<size>            Spatial blocking size along 2nd dimension
                              [default: 32]
      --bs3=<size>            Spatial blocking size along 3rd dimension
                              [default: -1]
      --sl=<size>             Streaming length for each block [default: 128]
      --ds                    Disable streaming computation [default: no]
      --nakata                Enable Nakata's register scheduling
                              [default: no]
      --sm_vec                Enable vector loading for shared memory
                              [default: no]
      --opt=<algorithm>       Select optimization method
                              (nondiag|assoc|none)
```

Each benchmark accepts following options:

```
./a.out [OPTION] ...

-s	Specify compute size
-t	Specify the number of timesteps
-n	Specify the number of executions
-c	Enable comparison to CPU execution
-h	Show this usage
```

## Performance Model

Model_P100.xlsm and Model_V100.xlsm are macro-enabled Excel sheets which implement our performance model. Parameter search for one stencil can be done using the "Parameter Sweep" button in the "Formula" sheet and the "Extract Top" button on the same sheet will automatically extract the top 5 configurations for all stencils. The "csv" files in the compile folder of the repository are generated in this way.
