# try to convert `prefixScan_t` from CUDA to OneAPI

### compile command
1. compile without-cub version

    ```
    make build
    ```
### run command
2. run without-cub version. Default: GPU device and zero-level backend

    ```
    make run
    ```
3. run without-cub version on CPU

    ```
	make run_cpu
    ```
4. run without-cub version on openCL backend

    ```
	make run_ocl
    ```
### clean project
5. remove ```.exe``` file

    ```
    make clean
    ```
    