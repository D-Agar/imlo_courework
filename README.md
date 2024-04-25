# IMLO Coursework Repository
## The Virtual Environment
Use the terminal for the following steps to create the virtual environment:
1. Create the environment from the `environment.yml` file:
    ```commandline
    conda env create -f environment.yml
    ```
2. Activate the new environment:
    ```commandline
    conda activate coursework
    ```
3. Check the environment was installed successfully:
    ```commandline
    conda env list
    ```
### CUDA or CPU
Ensure you check whether your system is CUDA-capable, as the environment assumes you are.
For more information check PyTorch's [website](https://pytorch.org/get-started/locally/).