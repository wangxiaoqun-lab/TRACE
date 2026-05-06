#!/bin/bash
#SBATCH --job-name=cell_mapping
#SBATCH --array=0-119
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu1
#SBATCH --output=cell_mapping.out
#SBATCH --error=cell_mapping.err

# Activate conda environment
source activate spateo

cd /OT_mapping/

# Define parameter spaces
ALPHAS=(0 0.001 0.1 0.3 0.5)
SLICE_NAMES=('CGE' 'LMGE')
TIME_POINTS=(0 1 2 3 4 5 6 7 8 9 10 11)

# Calculate dimensions
NUM_ALPHAS=${#ALPHAS[@]}
NUM_SLICES=${#SLICE_NAMES[@]}
NUM_TIME_POINTS=${#TIME_POINTS[@]}

# Map array index to parameters
TASK_ID=$SLURM_ARRAY_TASK_ID

# Calculate indices
time_point_idx=$((TASK_ID / (NUM_ALPHAS * NUM_SLICES)))
remaining=$((TASK_ID % (NUM_ALPHAS * NUM_SLICES)))
alpha_idx=$((remaining / NUM_SLICES))
slice_idx=$((remaining % NUM_SLICES))

# Get parameter values
ALPHA=${ALPHAS[$alpha_idx]}
SLICE_NAME=${SLICE_NAMES[$slice_idx]}
TIME_POINT_IDX=${TIME_POINTS[$time_point_idx]}

# Run Python script with all parameters
python main_mappings.py $ALPHA $SLICE_NAME $TIME_POINT_IDX

echo "Job $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID: \
TimePoint=${TIME_POINT_IDX} (${time_points[TIME_POINT_IDX]}), \
Alpha=$ALPHA, Slice=$SLICE_NAME"