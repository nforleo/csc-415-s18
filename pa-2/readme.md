Before compilation:
```bash
$ module load LibTIFF
```

Can compile with:
```bash
$ g++ -std=c++11 -Wall -ltiff conv2d.cc -o prog
```

And run with:
```bash
$ sbatch myjob.sh
```
