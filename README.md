# RHE-mc
Randomized Heritability Estimation for Multi-variance Components



## Prerequisites
The following packages are required on a linux machine to compile and use the software package.
```
g++ (4.4.7)
cmake
make
```

## How to install :

```
git clone https://github.com/alipazokit/RHE-mc.git
cd RHE-mc
mkdir build
cd build/
cmake ..
make
```

# Documentation for RHE-mc
An executable file named RHEmc will be in build folder after the installation steps. Run RHE-mc as follows:
 ```
 ./RHEmc <command_line arguments>
```
To estimate dominance heritability and additve heritability jointly run :

```
 ./RHEmc_dom <command_line arguments>
```

## Parameters

```
genotype (-g) : The path of  genotype file
phenotype (-p): The path of phenotype file
covariate (-c): The path of covariate file
annotation (-annot): The path of annotation file.
num_vec (-k) : The number of random vectors (10 is recommended). 
num_block (-jn): The number of jackknife blocks. (100 or 22 are recommended). The higher number of jackknife blocks the higher memory usage.
out_put (-o): The path of output file.


```
## File formats
```
Genotype : It must be in bed format.
Phenotype: It must have a header in the following format: FID IID name_of_phenotype
Covariate: It must have a header in the following format: FID IID name_of_cov_1 name_of_cov_2  . . .   name_of_cov_n
Annotation: It has M rows (M=number  of SNPs) and K columns (K=number of annotations). If SNP i belongs to annotation j, then there is  "1" in row i and column j. Otherwise there is "0". (delimiter is " ")

1) Number and order of individuals must be same in phenotype, gentype and covariate files.
2) Number and order of SNPs must be same in bim file and annotation file.
3) Annotation file does not have a header. The code supports overlapping annotations (e.g : functional annotation)
4) SNPs with MAF=0 must be excluded from the genotype file.
5) RHE-mc excludes individuals with NA values in the phenotype file from the analysis.
```
## Toy example 
To make sure that everything works well, sample files are provided in example directory. Look at test.sh file and run it  :
```

chmod +x test.sh
./test.sh

```

## Citation
```
Pazokitoroudi, A., Wu, Y., Burch, K.S. et al. Efficient variance components analysis across millions of genomes. Nat Commun 11, 4020 (2020). https://doi.org/10.1038/s41467-020-17576-9
```


