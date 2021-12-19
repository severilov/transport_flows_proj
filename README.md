# transport_flows_proj

## Installation

1. Clone this repository.

    ```bash
    git clone https://github.com/severilov/transport_flows_proj.git
    cd transport_flows_proj
    ```

2. Create conda environment.

    ```bash
    conda env create -f vladik.yml
    conda activate vladik
    ```

3. (Recommended) Run script at least one time to initialize all results folders. Script should be executed from the root directory of project.

    ```bash
    (vladik) python3 code/multi-stage-new.py base
    ```

## Accelerated Sinkhorn

To get results using accelerated version of Sinkhorn algorithm, simply
run script multi-stage-new.py with correct flag. Remember that the script shoud be executed from the root directory of project:

    ```bash
    (vladik) python3 code/multi-stage-new.py accelerated
    ```
