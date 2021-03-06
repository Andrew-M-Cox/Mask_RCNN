#!/bin/bash
#SBATCH --partition=GPUv100s                                  # select partion GPU
#SBATCH --nodes=1                                        # number of nodes requested by user
#SBATCH --gres=gpu:1                                 # use generic resource GPU, format: --gres=gpu:[n], n is the number of GPU card
#SBATCH --time=5-00:00:00                                # run time, format: D-H:M:S (max wallclock time)

source activate mrcnn3
cd
cd /project/bioinformatics/Kim_lab/s434049/Mask_RCNN/samples/chromosome/
python chromosome.py train --dataset=/project/bioinformatics/Kim_lab/s434049/Mask_RCNN/datasets/chrom_synthetic --weights='coco'
