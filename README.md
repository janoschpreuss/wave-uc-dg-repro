# wave-uc-dg-repro

This repository contains software and instructions to reproduce the numerical experiments in the paper
> Unique continuation for the wave equation based on a discontinuous Galerkin time discretization
> * Erik Burman and Janosch Preuss
> * University College London

# How to run / install
We describe two options to setup the software for running the experiments.

* downloading a `docker image` from `Zenodo` or `Docker Hub` which contains all dependencies and tools to run the application,
* or installing the required software in a `conda` environment. 

## Docker image 

### Pulling the docker image from Docker Hub 
* Please install the `docker` platform for your distribution as described [here](https://docs.docker.com/get-docker/).
* After installation the `Docker daemon` has to be started. This can either be done on boot or manually. In most Linux 
distributions the command for the latter is either `sudo systemctl start docker` or `sudo service docker start`.
* Pull the docker image using the command `docker pull janosch2888/wave-uc-dg-repro:v1`. 
* Run the image with `sudo docker run -it janosch2888/wave-uc-dg-repro:v1 bash`.
* Proceed further as described in [How to reproduce](#repro).

### Downloading the docker image from Zenodo
* For this option the first two steps are the same as above.
* The image can be downloaded [here]( ). 
* Assuming that `wave-uc-dg-repro.tar` is the filename of the downloaded image, please load the image with `sudo docker load < wave-uc-dg-repro.tar`.
* Run the image with `sudo docker run -it janosch2888/wave-uc-dg-repro:v1 bash`.
* Proceed further as described in [How to reproduce](#repro).

## Installing in a conda environment
* Please install `conda` as described in detail [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
* Then download the file `wave-uc-dg-repro.yml` from `Zenodo` or from this `github` repository.
* Open a bash shell in the folder that contains this file and execute (in the conda base shell) 

    conda env create -f wave-uc-dg-repro.yml
    conda activate wave-uc-dg-repro 

Then clone the repository 

    git clone https://github.com/janoschpreuss/wave-uc-dg-repro.git

and change into the folder `wave-uc-dg-repro`. Now we can proceed further as described in [How to reproduce](#repro).


# <a name="repro"></a> How to reproduce
The `python` scripts for runnings the numerical experiments are located in the folder `scripts`.
To run an experiment we change to this folder and run the corresponding file.
After execution has finished the produced data will be available in the folder `data`.
For the purpose of comparison, the folder `data_ref` contains a copy of the data which has been used for the plots in the paper.


To generate the plots as shown in the article from the data just produced we change to the folder `plots`
and compile the corresponding `latex` file.
Below we decribe the above process for each of the figures in the article in detail.
For viewing the generated pdf file, say `figure.pdf`, the figure has to be copied to the host machine.
This can be done by executing the following commands in a new terminal window (not the one in which `docker` is run):

    CONTAINER_ID=$(sudo docker ps -alq)
    sudo docker cp $CONTAINER_ID:/home/app/wave-uc-dg-repro/plots/figure.pdf \
    /path/on/host/machine/figure.pdf

Here, `/path/on/host/machine/` has to be adapted according to the file structure on the host machine.
The file `figure.pdf` can then be found at the designated path on the host machine and inspected with a common pdf viewer.
(The command above assumes that the reproduction image is the latest docker image to be started on the machine).
Alternatively, if a recent latex distribution is available on the host machine it is also possible to copy data and tex files to the latter and
compile the figures there.

## <a name="Fig1"></a> Figure 1
Change to directory `scripts`. Run 
    
    python3 precond-1d-comp.py 1
    python3 precond-1d-comp.py 2

The following data files will be produced:
* `precond-1d-iters-order__j__.dat` where __j__ in [1,2] denotes the finite element order. This table contains the GMRes iteration numbers for the different methods. 
The columns `DFB` contains results for the decoupled forward-backward solve, the columns `MTM-full` and `MTM-lo` for the monolitic time-marching with full and minimal 
dual order, respectively. The column `block` is for the Block-Jacobi preconditioner and the column `vanilla` without any preconditioner.
* The files `precond-1d-L2L2ut-order__j__.dat` contain the L2L2-errors for the different methods, whereas the files `precond-1d-LinftyL2u-order__j__.dat` contain the 
L-infinity(time)-L2(space) errors. As above, __j__ in [1,2] gives the order of the FEM.  
* The files `precond-1d-DFB-order2-residuals-ref-lvl__i__.dat` contain the GMRes residuals for the DFB method for __i__ in [2,3,4]. Here, __i__=3 corresponds to N=8 and __i__=4 to N=16. The files `precond-1d-MTM-lo-order2-residuals-ref-lvl__i__.dat` contain the corresponding residuals for the monolitic time-marching method with minimal dual stabilization.

To generate Figure 1, switch to the folder `plots` and run 
 
    latexmk -pdf Figure1.tex

## <a name="Fig2"></a> Figure 2 and Table 1
Change to directory `scripts`. Run 

    python3 precond-3d-comp.py 1 1
    python3 precond-3d-comp.py 2 1
    python3 precond-3d-comp.py 3 1

The following data files will be produced:
* The files `precond-3d-GCC-iters-order__j__.dat` where __j__ in [1,2,3] denotes the order of the FEM contain the iteration numbers and specific information about the degrees of freedom (dof) as given in Table 1 of the paper. The columns `ndof-tot-DFB`,`ndof-inv-DFB` and `DFB-iter` contain the total number of dofs, number of dofs in the linear system to be inverted and the iteration numbers for the DFB method. The columns `ndof-tot-MTM-lo`,`ndof-inv-MTM-lo` and `MTM-lo-iter` contain the same quantities for the monolitic time-marching. 
* The files `precond-3d-GCC-L2L2ut-order__j__.dat` contain the L2L2-errors and the files `precond-3d-GCC-LinftyL2u-order__j__.dat` the L-infinity-L2 errors for both methods. The column `deltat` denotes the size of the time step.
* The vtk data of the absolute error shown in the center of Figure 2 is contained in the file `abserr-cube-GCC-order1-MTM-lo-u.xdmf`. 

To generate Figure 2 and Table 1, switch to the folder `plots` and run 

    latexmk -pdf Figure2.tex
    latexmk -pdf Table1.tex

## <a name="Fig3"></a> Figure 3
Change to directory `scripts`. Run 

    python3 precond-3d-comp.py 1 0
    python3 precond-3d-comp.py 2 0
    python3 precond-3d-comp.py 3 0

The generated data files are called exactly the same way (and have the same structure) as the ones in [Figure 2](#Fig2) except for `GCC`being replaced by `noGCC`.
To generate Figure 3, switch to the folder `plots` and run 

    latexmk -pdf Figure3.tex


## <a name="Fig4"></a> Figure 4
Change to directory `scripts`.
 
    python3 noGCC-restricted-1d.py

Data files of the form `noGCC-restricted-1d-order__j__.dat` for __j__ in [1,2,3] representing the order of the FEM will be created. The columns `LinftyL2u-all` 
and `L2L2ut-all` represent the L-infinity-L2 and L2-L2 errors in the whole space-time domain, whereas the columns `LinftyL2u-restrict` and `L2L2ut-restrict` 
contain the errors in the restricted set B_t (shown as dashed lines in the plot). The plot of the absolute error shown in the middle of Figure 4 is available 
in the file `abs-err.png`.

To generate Figure 4 switch to the folder `plots` and run 

    latexmk -pdf Figure4.tex


