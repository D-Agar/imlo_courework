# IMLO Coursework Repository
## The Virtual Environment
Use the terminal for the following steps to create the virtual environment. assuming you are using `conda`:
1. Create the environment from the `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    ```
2. Activate the new environment:
    ```bash
    conda activate coursework
    ```
3. Check the environment was installed successfully:
    ```bash
    conda env list
    ```
### CUDA or CPU
Ensure you check whether your system is CUDA-capable, as the environment assumes you are.
For more information check PyTorch's [website](https://pytorch.org/get-started/locally/).
## Running the file
```shell
python imlo_model.py
```