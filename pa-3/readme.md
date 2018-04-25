Before compilation:
```bash
$ module load LibTIFF
$ module load CUDA
```

The demo file can be compiled with:
```bash
$ nvcc conv-1d-gpu.cu -std=c++11 -o prog
```

And run with:
```bash
$ sbatch myjob.sh
```
