# 🎓 Tutorial: Automated Distributed ML Pipelines with DVC and Ray

This tutorial will guide users through creating automated, scalable, and distributed ML pipelines using DVC (Data Version Control) integrated with Ray. 

- Run Distributed ML Pipeline with DVC [DVC](https://dvc.org/). 
- Design distributed ML pipeliens with Ray [Ray](https://www.ray.io/). 
- Introduce [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) for hyperparameter optimization
- Run distributed ML experiment with DVC and Ray on AWS

![DVC + Ray](src/static/preview.png "DVC + Ray")

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run DVC pipeline

```bash
export PYTHONPATH=$PWD
dvc exp run
```

## Run Ray Cluster (local)

Locally: cpu, single-node

```bash
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1
ray start --head --port=6379
```

Run DVC pipeline

```bash
dvc exp run
```

### Examples

Submit Ray job

```bash
ray job submit --working-dir . -- python src/train.py
```

## Run Ray Cluster at AWS

Example: <https://docs.ray.io/en/latest/cluster/vms/examples/ml-example.html#clusters-vm-ml-example>

## Deploy a Ray cluster

```bash
# Run a cluster
ray up cluster.yaml  
```

### Running Jobs Interactively
If you would like to run an application interactively and see the output in real time (for example, during development or debugging), you can кun your script directly on a cluster node (e.g. after SSHing into the node using `ray attach`

```bash
# Attach to ssha
ray attach cluster.yaml

# Exec shell commands
ray exec cluster.yaml 'echo "hello world"'
```

### Open Ray Dashboard

```bash
ray dashboard cluster.yaml
```

### Submit a job 

The ray submit and ray job submit commands are used in the Ray framework for different purposes.

#### `ray submit`
ray submit is used to submit a Python script to a Ray cluster. The script is run on the head node of the cluster. This command is useful when you want to run a script on a Ray cluster without having to SSH into the head node and run the script manually.

```bash
ray submit cluster.yaml src/mnist.py
```

#### `ray job submit`
ray job submit, on the other hand, is part of Ray's job submission API, which is a higher-level API for running jobs on a Ray cluster. This command submits a job to the Ray cluster, where a job is defined as a Python script along with the necessary environment and dependencies. The job is run on the cluster and the results are returned when the job is finished.

To tell the Ray Jobs CLI how to find your Ray Cluster, we will pass the Ray Dashboard address. This can be done by setting the RAY_ADDRESS environment variable ([docs](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/quickstart.html?highlight=export%20address#submitting-a-job)):

```bash

export RAY_ADDRESS='http://localhost:8265'
ray job submit --working-dir $PWD -- python src/mnist.py --config params.yaml
```

Test:
```bash
ray job submit --working-dir $PWD -- python src/mnist.py
ray job submit --working-dir $PWD -- python src/stages/tune.py
ray job submit --working-dir $PWD -- dvc exp run
```

### Stop Cluster

```bash
ray down .dev/cluster/demo.yaml 
```


### Run DVC pipeline on Ray Cluster (remote)

```bash
ray exec cluster.yaml "git clone https://github.com/mnrozhkov/tutorial-mnist-dvc-ray.git"
ray exec cluster.yaml "cd tutorial-mnist-dvc-ray && \
    export PYTHONPATH=/home/ray/tutorial-mnist-dvc-ray && \
    dvc exp run"
ray exec cluster.yaml "cd tutorial-mnist-dvc-ray && \
    export PYTHONPATH=/home/ray/tutorial-mnist-dvc-ray && \
    dvc exp run"
```

