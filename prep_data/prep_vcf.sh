#!/bin/bash
​
INFOLDER=/bigdata/smathieson/pg-gan/1000g/ALL/
OUTFOLDER=/bigdata/smathieson/pg-gan/1000g/VCF/
SUFFIX=.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz
​
# NOTE: I commented out the actual commands, so when you run this the first time you can see the command printed
​
# for each set of population(s)
for POP in CHS # TODO change to your population
do
​
 # for each chromosome
 for CHROM in `seq 1 22`
 do
     echo "bcftools view -S ${POP}_gan.txt --min-ac 1:minor -m2 -M2 -v snps -Oz -o ${OUTFOLDER}${POP}.chr${CHROM}${SUFFIX} ${INFOLDER}ALL.chr${CHROM}${SUFFIX}"
     bcftools view -S ${POP}_gan.txt --min-ac 1:minor -m2 -M2 -v snps -Oz -o ${OUTFOLDER}${POP}.chr${CHROM}${SUFFIX} ${INFOLDER}ALL.chr${CHROM}${SUFFIX}
 done
​
 # then merge into one vcf
 echo "bcftools concat -f ${POP}_filelist.txt -Oz -o ${OUTFOLDER}${POP}${SUFFIX}"
 bcftools concat -f ${POP}_filelist.txt -Oz -o ${OUTFOLDER}${POP}${SUFFIX}
done
