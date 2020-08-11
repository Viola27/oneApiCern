# try to convert `prefixScan_t` from CUDA to OneAPI

### compile command
1. compile two version (with and without cub)

    ```
    make all
    ```
2. compile cub version 

    ```
    make build_cub
    ```
3. compile version without cub

    ```
    make build
    ```
4. compile with link option for assertion

    ```
    make buid_l
    ```
### run command
5. run cub version 

    ```
    make run_cub
    ```
6. run version without cub

    ```
    make run
    ```
### clean project
7. remove ```.exe``` file

    ```
    make clean
    ```
    