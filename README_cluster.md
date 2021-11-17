# Manual for compute cluster at LNT

## Introduction (german)
[cluster_einfuehrung.pdf](cluster_einfuehrung.pdf)

## Most important commands
* `qsub -cwd myscript.sh`
* `qsub_matlab mymatlabscript var1=[1:2:10] var2=linspace(1,10,3)`
* `qstat -f`
* `cluster_users.py`
* `qdel 123456`
* `qdel -u username`
* `qrsh -now n`

## Tutorial with useful commands and hints for lntcluster
*lntcluster* is the *front-node*, which distributes the computation jobs to several *compute-nodes*  (`compute-x-x`) using the *gridengine*
This tutorial assumes you are on a linux shell. On Windows, you can use putty. Commands starting with `... ` are to be used in the command line.

### Login and Logout on the front-node 
You need to login to the front-node in order to submit jobs, look up which jobs are already done, ... No computations are allowed to be run on the front-node.
* Login via `ssh lntcluster`. It will ask for your password and uses your local user name. (Alternatively use `ssh username@lntcluster` to provide the username explicitly). 
* you can login without entering the password every time. create a ssh-key first and copying it to the lntcluster using something like `ssh-copy-id -i ~/.ssh/id_rsa.pub lntcluster`. [full manual (external)](https://www.thegeekstuff.com/2008/11/3-steps-to-perform-ssh-login-without-password-using-ssh-keygen-ssh-copy-id)
* you may leave the lntcluster with `exit` (optional)
### Login and Logout on the compute-nodes 
Sometimes, you may need to login on the actual compute nodes. For example, you can check the ressource usage using `top` or debug something. Though it is technically possible, it is not allowed to perform computations that way, cause it circumvents the job-queue.
* To login on a compute-node, first login to lntcluster as described above.
* Now, type `ssh compute-0-0` to login into the node 0-0. You may need to enter your password again. You may verify that you are on the compute-0-0 node using `hostname`.
* Run `top` to get the currently used resources. Leave *top* by pressing `q`.
* Logout from the node with `exit`. You are now back on the lntcluster.


### Status of cluster and your own jobs
You may want to know the status of the cluster (How many jobs are running there, how busy is it?) and the status of your own jobs (How many jobs are still running? How much resources are they using?). There are several possibilities to accomplish this.
* Visit `http://lntcluster/ganglia/` in your browser. It will provide you a nice overview of the overall cluster resources. You can also look at specific nodes.
* `cluster_users.py` [cluster_users.py](Skripte/cluster_users.py) to check who is running how many jobs at the moment and how many jobs are in the queue.
* `cluster_statistic.py 10` [cluster_statistic.py](Skripte/cluster_statistic.py) showns the CPU-seconds of all users for the last 10 days. It also calculated CPU-weeks, energy costs and CO2 usage for convenience.
* `qstat -f` provides a list of your own jobs and tells you on which node they are running. Additionally, it shows a short summary of what is going on the node. Example:  
  ![screenshow_qstat-f](Images/screenshow_qstat-f.png)  
Here, 4 jobs of user *grosche* are running on *compute-0-1*. This node cannot take more jobs since 20/20 are already in use. Node *compute-0-0* may take 7 more jobs. In both cases, the load_avg is higher than the number of jobs, i.e., that some of the jobs actually request slightly more than one core and some computations are being queued on the processor. Furthermore, we can see that compute-0-4 is currently down or not accessible.

* We might want to get more information about the first job with id *3741899*. This can be done using `qstat -j 3741899`. It will tell you the working directory, the fully name of the exectued script and more information that may be useful for debugging.
* To get some information about jobs that have already finished, use  `qacct -j 3741899`.


### Submitting Jobs
Jobs are submitted to the cluster using `qsub example.sh`. Only shell scripts can be submitted directly. One either needs to write a shell script which then calls the actual program (done for the python code, see below) or uses some convenient scripts that are available, too. In any cases, one has to ensure that the program does not use too many resources, see [best practices and limits (below)](#best-practices-and-limits).

#### qsub_matlab
* Submit a matlab-skript to the cluster: ``qsub_matlab <matlabscript>``
    * The scriptname is necessary here, without the trailing *.m* of the filename! The matlab script is copied to a version where date and time are appended. A shellscript will be generated automatically. 
    * Example: ``qsub_matlab my_special_matlab_script``
* Submit a set of matlab evaluations to the cluster: ``qsub_matlab <matlabscript> <variable=matlab_vector> <variable=matlab_vector>``
    * The scriptname is necessary here, without the trailing *.m* of the filename! The matlab script is copied to a version where date and time are appended.
    * Example: ``qsub_matlab my_special_matlab_script 'para1=[1:2:10]' 'para2=linspace(1,10,3)'``
    * In this case every combination of para1 and para2 is evaluated (1,1) (1,5.5) (1,10), (3,1) (3,5.5) (3,10)... The script *qsub_matlab* shows the number of evaluation and suggests to call the generated *qsub_start_eval.sh* if sure. The script *./qsub_start_eval.sh* generates for each evaluation 1 shellscript and 1 matlab-script. The matlab-script looks like ``para1=5 para2=10 my_special_matlab_script``
    * The shellscript (and therefore the matlab script) is automatically submitted to the cluster. Of course *para1* and *para2* are only examples and you may specify as many *<variable=matlab_vector>* combinations as you need. 
 *  In order to need less licenses, one could also compile matlab code with something like `mcc -m -R '-singleCompThread' MySimulation.m`

#### qsub_binary (not tested in 2019, is anyone using this?)
* Submit a (set of) jobs consisting of a shellscript or a binary: ``qsub_binary <matlabscript> <variable=matlab_vector> <variable=matlab_vector> ...``
* For general description look at qsub_matlab
* Example: Example: ``qsub_binary binary_or_script 'para1=[1:2:10]' 'para2=linspace(1,10,3)'``
    * The script or binary *'binary_or_script'* is subsequently called with all combinations of *para1* and *para2* as in *'binary_or_script 3 1'*
    * Additionally the corresponding environment variables are set, in this example *para1* and *para2*.

#### qsub anything (here, python)
You can start a python script (or any other program) with some parameters using a combination of two shell files. One shell file is used to loop through the parameters and submit one job for each set of parameters. The second shellfile is executed using these parameters.
A minimal example is located in  the folder [Skripte/qsub_python_minimal](Skripte/qsub_python_minimal)
It already includes setting the relevant environment variables to make python (at least with numpy, scipy, .. this should work) use only one core are:
```
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1  
export OMP_NUM_THREADS=1
```
To run the jobs, execute  `$sh shellscript_qsub.sh`

The most important options for qsub are
* `qsub -cwd` use the current working directory as working directory of the job
* `qsub  -v p1="${MYPAR1}",p2="${MYPAR1}" shellscript_pythoncall.sh ` pass local variables to the shell script which can then use those to pass them to the actual program to be executed.
* `qsub -q all.q@compute-1-1.local` only use node compute-1-1. This may be useful

### Deleting Jobs
* `qdel 123456` deletes job with specific id
* `qdel {12341..123415}` deletes the jobs 12341, 12342, 12343, 12344, 12345
* `qdel -u grosche` deletes all jobs from user *grosche*.

### Best practices and limits
Often, there is some trade-off between few jobs or many jobs. Here is general advise
* Don't use ultra-fast jobs, since there is some overhead (several seconds) from the job-scheduler.
* Don't use long-running jobs. Don't run 300 jobs that last several weeks. Cut it into smaller pieces, if possible
* Everything between some hours and a few days is just fine. 
* Current maximum number of jobs is 10000.
* The number of nodes on the cluster is currently 616. This is the maximal number of jobs. The job scheduling may leave some nodes unused in order to allow other users that come later to also run there jobs.

There are some limitations for the resources of the individual jobs. These limits need to be handled manually by the user. There are no strict restrictions of the scheduler at the moment, since this a lot of work and less flexible. This means, that a job could in principle use all cores and the whole memory of a node. However, this will eventually kill the node when more jobs are launched on it. Additionally, it is not fair for other jobs when some jobs request more resources. The current limits your jobs need to satisfy are: 
* Each job may only use 1 processor core.
* Each job may only use roughly 5-6 GB of memory, dependent on the node.
* Not too much (in terms of number and size) file input-output from the hard drives.
* (Not too many opened files (e.g. from matlabs addpath/genpath commands or own libraries such as anaconda).)
* Do not use too many licenses. For matlab code, you could pre-compile code. Licenses are counted per user per node, so it is also a possibility to restrict ones own jobs to new nodes.

How to ensure that your jobs satisfy the limits?
* Using `qsub_matlab` automatically makes matlab use only one core via that `-singleCompThread` option.
* First run locally and check the needed resources.
* Check locally that your job only needs one core.
* For some libraries (including most python libraries), you can export environment variables (see job submission section or (shellscript_pythoncall.sh)[Skripte/qsub_python_minimal/shellscript_pythoncall.sh]).
* Check locally that your job does not need too much memory
* When everything is fine locally, run only few jobs first. Monitor them with `qstat -f` and ssh compute-x-x  and `top`.
* When few jobs run nicely, run more jobs, keep monitoring from time to time.
* If something breaks, check locally.

### Debugging
You may need to debug a job for various reasons or just wish to do some quick calculations (e.g., also for debugging) on the cluster environment. You may not run computations on the front-node or a compute-node directly via ssh, so you need to enter an interactive job. This can be done using `qrsh -now n`. It will try to get a job for you and you end up in a shell on one of the compute nodes if it can launch a new job. Verify by checking the output of `hostname`. 
If you check the output of `qstat -f` (in another terminal), you notice that your qrsh-shell is just another job with a normal job-id and name *QRLOGIN*. Limitations and restrictions (only one core!) are the same as for any other job.
`exit` from the qrsh when you are done.

### Python installation
In order to use Python on the Cluster (and locally) it is highly advised to use [Anaconda (external)](https://www.anaconda.com/distribution/). Just install it on the cluster (from the front-node). Create an environment with *conda* and use this environment as explained in the example **link to example**.

### Minor hints, known bugs
* Some software may have memory leaks that only arise after calling a function million of times. Such memory leaks may not arise on your local machine running only ten parameters (in one script) but may arise when you let the cluster run the script for a long time for millions of parameters.
* qrsh used different PATH environment variables than normal qsub-jobs some time ago, better check it if you feel that something is missing in your PATH
* /dev/shm/ may be used to temporarily write files that are later deleted. It is not located on the hard drive but on the memory. Of course, this counts to your used memory. Make sure to delete the files afterwards, make sure to make the filename unique for each job. If some job did not finish and therefore the temporary files remain, (Skripte/clear_own_devshm_files)[Skripte/clear_own_devshm_files] provides scripts to clear those remaining files.
* found a Bug? Add it here!