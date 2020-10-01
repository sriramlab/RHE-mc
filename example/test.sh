
gen=1
phen=small.pheno.plink
covar=sample.cov
annot=sample.annot.txt

../build/RHEmc -g $gen -p $phen -c $covar -k 10 -jn 100   -o test.out.txt -annot $annot






