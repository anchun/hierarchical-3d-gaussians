#!/bin/bash
scenes=("${HOME}/data/wayve/scene_081")
for DATASET_DIR in "${scenes[@]}"; do
    echo "Processing dataset: ${DATASET_DIR}"
    startTime_s=`date +%s`

    # python preprocess/undistort_wayve.py --project_dir ${DATASET_DIR}
    python preprocess/generate_pose_prior.py --project_dir ${DATASET_DIR}
    python preprocess/generate_chunks.py --project_dir ${DATASET_DIR} --skip_bundle_adjustment
    python preprocess/generate_depth.py --project_dir ${DATASET_DIR}
    python scripts/full_train.py --project_dir ${DATASET_DIR} --chunks_iterations 30000 --skip_if_exists

    endTime_s=`date +%s`
    elapsedTime_s=$[ $endTime_s - $startTime_s ]
    echo "Processing completed in ${elapsedTime_s} seconds."
    echo "----------------------------------------"
done

