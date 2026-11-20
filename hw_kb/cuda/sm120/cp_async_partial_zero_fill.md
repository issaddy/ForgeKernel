# CUDA C++ partial cp.async zero fill

## Purpose

Stage a 16-byte aligned fragment with a runtime valid-byte count. Inline PTX takes a 32-bit shared address, a 64-bit global pointer, and the valid byte count. The remainder is zero-filled. Commit and wait on the issuing thread, then rendezvous before reading another thread's writes.

## Minimal runnable example

Run the [CUDA probe](probes/async_copy_and_warp_sum.cu) on SM120:

```sh
nvcc -arch=sm_120a -O3 async_copy_and_warp_sum.cu -o cpp_sm120
./cpp_sm120
cuobjdump --dump-sass cpp_sm120 | grep LDGSTS
```

## Gotchas

The probe requests 0, 4, 8 or 12 bytes per lane from valid aligned source addresses. Convert the destination with `__cvta_generic_to_shared` and wait before reading the result.

## Sources

- [PTX ISA, cp.async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async).
- CUDA 13.0.88 nvcc on RTX PRO 5000 72GB Blackwell: `PROBE PASS max_diff=0.0` for all partial copies.
