# Working the Goethe Uni HLR Cluster

## Identify Cluster Hardware and Partitions

Use  `scontrol show nodes | grep "Partitions=gpu" -B 10` to go through the nodes available and their hardware specifications. Look at the example output beneath. As can be observed the NVIDIA GPUs are in partition gpu2 while most nodes are gpu partitions nodes that run AMD. Going through the HLR website you can find that they specifically use the AMD Radeon 
Instinct MI210 GPUs.

```
NodeName=gpu31-021 Arch=x86_64 CoresPerSocket=64 
   CPUAlloc=0 CPUTot=256 CPULoad=0.01
   AvailableFeatures=NVIDIA,RACK31
   ActiveFeatures=NVIDIA,RACK31
   Gres=gpu:2
   NodeAddr=gpu31-021 NodeHostName=gpu31-021 Version=18.08
   OS=Linux 3.10.0-1160.83.1.el7.x86_64 #1 SMP Tue Jan 24 08:34:19 CST 2023 
   RealMemory=1900000 AllocMem=0 FreeMem=2024055 Sockets=2 Boards=1
   MemSpecLimit=2048
   State=IDLE ThreadsPerCore=2 TmpDisk=1400000 Weight=10 Owner=N/A MCS_label=N/A
   Partitions=gpu2 
--
NodeName=gpu35-001 Arch=x86_64 CoresPerSocket=32 
   CPUAlloc=64 CPUTot=64 CPULoad=0.53
   AvailableFeatures=GPU,Rack35
   ActiveFeatures=GPU,Rack35
   Gres=(null)
   NodeAddr=gpu35-001 NodeHostName=gpu35-001 Version=18.08
   OS=Linux 5.14.0-162.23.1.el9_1.x86_64 #1 SMP PREEMPT_DYNAMIC Tue Apr 11 10:43:28 EDT 2023 
   RealMemory=500000 AllocMem=497952 FreeMem=501440 Sockets=2 Boards=1
   MemSpecLimit=2048
   State=ALLOCATED ThreadsPerCore=1 TmpDisk=700000 Weight=10 Owner=N/A MCS_label=N/A
   Partitions=gpu 
--
NodeName=gpu35-002 Arch=x86_64 CoresPerSocket=32 
   CPUAlloc=64 CPUTot=64 CPULoad=3.05
   AvailableFeatures=GPU,Rack35
   ActiveFeatures=GPU,Rack35
   Gres=(null)
   NodeAddr=gpu35-002 NodeHostName=gpu35-002 Version=18.08
   OS=Linux 5.14.0-162.23.1.el9_1.x86_64 #1 SMP PREEMPT_DYNAMIC Tue Apr 11 10:43:28 EDT 2023 
   RealMemory=500000 AllocMem=497952 FreeMem=481911 Sockets=2 Boards=1
   MemSpecLimit=2048
   State=ALLOCATED ThreadsPerCore=1 TmpDisk=700000 Weight=10 Owner=N/A MCS_label=N/A
   Partitions=gpu 
```
## Set up your environment

### Downloading miniconda
Miniconda is a free minimal installer for conda. It is a small, bootstrap version of Anaconda that includes only conda, Python, the packages they depend on, and a small number of other useful packages, including pip, zlib and a few others.  If conda is not set up on your system (verify via `conda -V`) it is recommended you install miniconda. Use the following `miniconda_install.sh` file or run the commands one by one.

```
echo "Downloading miniconda..."
# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh


echo "Installing miniconda..."
# Install Miniconda
bash miniconda.sh -b -p $HOME/miniconda
rm miniconda.sh
```

run `conda list python -f` to verify your installation

### Prepare your environment

First make a new conda env `conda create -n myenv python=3.9
` and activate it via `conda activate myenv`. As we want to run on AMD GPUs we need to download the torch ROCm library, you can find the command at the [offecial pytorch website](https://pytorch.org/get-started/locally/). It will look something like:

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
```
## Set up your code

### python code
Slurm jobs allow you to run multiple jobs. Usually you will want to vary each run slightly. This can be achieved by passing an argument when running the functions. For instance, when running `python3 slurm_test.py Hello` the argument can be accessed in the python code:
 
```
> import sys
> print(sys.argv[1])
< Hello
```
If you are simply running a 10-fold validation you can pass 10 different numbers and set `random.seed(int(sys.argv[1]))` (remember to do the same for `np.random` if using numpy). This insures your results are reproducible. For more complicated control define your hyperparameter for each run in a dictionary and pass it as a `hyper_parameters.npy` file:

```
> hp = np.load('hyper_parameters.npy', allow_pickle=True)
> print(hp)
< hp = {  '1': {'lr':0.01, 'batch_size'=128, 'opt': 'SGD'},
< 	'2': {'lr':0.1, 'batch_size'=256, 'opt': 'ADAM'},
< 	...}
```

My git has [simple examples](https://github.com/sari-saba-sadiya/Feature-Imitating-Networks/blob/main/generate_topology.ipynb) of how to generate hyper_parameters.npy files. This random grid search method is sufficient for most use cases.

### The SLURM jobs

To run multiple jobs in parallel define a bash file `slurm_job.sbatch`



```
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --array=0-4:2
#SBATCH --mail-type=FAIL
#SBATCH --output out/output_%a.txt
#SBATCH --error err/error_%a.txt

# Add Miniconda to PATH
source $HOME/miniconda/etc/profile.d/conda.sh
conda init bash
conda activate pytorch
python3 slurm_test.py %a
```
The param `partition=gpu` instructs slurm to run only on nodes in the GPU partition, the `node` defines how many nodes to reserve in total, while `ntask` is the number of jobs that each of your python runs can execute in parallel. ignore this parameter unless your code is parallelized.`gres=gpu:1` sets the  number of GPUs per node. Note that even if you do not use this param your code will still execute on GPU. 


The most interesting argument for us is `#SBATCH --array=0-4:2` what this line does is run jobs with indexes between 0 and 4 with a step of 2. Meaning we will run three jobs where the argument received by `slurm_tets.py` is  equal to `0,2,4`. To simply run 10 jobs use  `#SBATCH --array=0-9`.

You will also need to `mkdir out; mkdir err` before running the command to save the output and error files from your run. Ofcourse, use `sys.argv[1]` in the name any file you are saving (model weights, results ...) to avoid conflicts between different runs. The command `#SBATCH --mail-type=FAIL` send you an email if your job fails.

# Running on SLURM

## sbatch and squeue

To run a slurm job use:

```
slurm_job.sbatch
```

To monitor your jobs use `squeue -u USERNAME` you can also keep track and update the job status using `watch -n 10 "squeue -u USERNAME" ` which updates the status every 10 seconds.


















