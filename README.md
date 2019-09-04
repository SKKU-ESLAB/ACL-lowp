# ACL-lowp
The project is designed to support efficient low precision computations on arm mobile devices.

It is implemented by utilizing SIMD MAC instructions of NEON for low precision data.

This library is not a typical GEMM but a GEMM library of bit-packed low precision data.

Currently, only general matrix multiplication(GEMM) for 4 bit data is supported.

## References
This project is implemented based on arm compute library

* arm compute library  [https://github.com/ARM-software/ComputeLibrary][acl]

## Prerequisites
* Scons

## Building
Build the arm computer library with scons.

Build command depends on the target architecture and the OS of the compilation environment.

For example, if the target architecture is armv7a and the compilation environment is linux, the command is

```
scons asserts=1 neon=1 examples=1 os=linux arch=armv7a
```

## Running the tests
There is example code executing low precision gemm

```
examples/neon_lowgemm.cpp
```

You can compile and run this by doing:

```
g++ examples/neon_lowgemm.cpp build/utils/*.o -O2 -std=c++11 -I. -Iinclude -Lbuild -larm_compute -larm_compute_core -larm_compute_graph -o neon_lowgemm
```
```
LD_LIBRARY_PATH=build/ ./neon_lowgemm
```

You can easily modify the example code and make your own.

[acl]:https://github.com/ARM-software/ComputeLibrary
