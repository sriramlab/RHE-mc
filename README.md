# RHE-mc
Randomized Heritability Estimation for Multi-variance Components



## Prerequisites
The following packages are required on a linux machine to compile and use the software package.
```
g++
cmake
make
```

## How to install :

```
git clone https://github.com/alipazokit/RHE-mc.git
cd RHE-mc
mkdir build
cmake ..
make
```

# Documentation for RHE-mc
An executable file named RHEmc will be in build folder after the installation steps. Run RHE-mc as follows:
 ```
 ./RHEmc <command_line arguments>
```
## Parameters

```
genotype (-g) : The path of a text file which contains pathes of genotypes files.
phenotype (-p): The path of phenotype file
covariate (-c): The path of covariate file
annotation (-annot): The path of annotation file.
num_vec (-k) : The number of random vectors (10 is recommended)
num_block (-jn): The number of jackknife blocks. (100 is recommended)
out_put (-o): The path of output file.

```




