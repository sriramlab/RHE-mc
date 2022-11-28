gen=small
phen=small.pheno
covar=small.cov
annot=annot.txt

../build/RHEmc -g $gen -p $phen -c $covar  -k 10   -jn 100  -annot $annot   -o test.out.txt






