#
# Copyright (C) 2024, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os, sys
import subprocess
import argparse
import time, platform
import shutil

def submit_job(slurm_args):
    """Submit a job using sbatch and return the job ID."""    
    try:
        result = subprocess.run(slurm_args, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error when submitting a job: {e}")
        sys.exit(1)
    
    job = result.stdout.strip().split()[-1]
    print(f"submitted job {job}")
    return job

def is_job_finished(job):
    """Check if the job has finished using sacct."""
    result = subprocess.run(['sacct', '-j', job, '--format=State', '--noheader', '--parsable2'], capture_output=True, text=True)
    
    job_state = result.stdout.split('\n')[0]
    return job_state if job_state in {'COMPLETED', 'FAILED', 'CANCELLED'} else ""

def setup_dirs(images, colmap, chunks, project):
    images_dir = os.path.join(project, "camera_calibration", "rectified", "images") if images == "" else images
    colmap_dir = os.path.join(project, "camera_calibration", "aligned") if colmap == "" else colmap
    chunks_dir = os.path.join(project, "camera_calibration") if chunks == "" else chunks

    return images_dir, colmap_dir, chunks_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', required=True, help="images, colmap and chunks paths doesnt have to be set if you generated the colmap using generate_colmap script.")
    parser.add_argument('--images_dir', default="")
    parser.add_argument('--global_colmap_dir', default="")
    parser.add_argument('--chunks_dir', default="")
    parser.add_argument('--chunk_size', default=100, type=float)
    parser.add_argument('--skip_bundle_adjustment', action="store_true", default=False)
    parser.add_argument('--keep_raw_chunks', action="store_true", default=False)
    parser.add_argument('--with_mvs', action="store_true", default=False)
    parser.add_argument('--n_jobs', type=int, default=8, help="Run per chunk COLMAP in parallel on the same machine. Does not handle multi GPU systems.")
    args = parser.parse_args()
    
    images_dir, colmap_dir, chunks_dir = setup_dirs(
        args.images_dir,
        args.global_colmap_dir, args.chunks_dir,
        args.project_dir
    ) 

    submitted_jobs = []

    start_time = time.time()

    ## First create raw_chunks, each chunk has its own colmap.
    print(f"chunking colmap from {colmap_dir} to {args.chunks_dir}/raw_chunks")
    make_chunk_args = [
            "python", f"preprocess/make_chunk.py",
            "--base_dir", os.path.join(colmap_dir, "sparse", "0"),
            "--images_dir", f"{images_dir}",
            "--chunk_size", f"{args.chunk_size}",
            "--output_path", f"{chunks_dir}/raw_chunks",
        ]
    try:
        subprocess.run(make_chunk_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing image_undistorter: {e}")
        sys.exit(1)

    ## Then we refine chunks with 2 rounds of bundle adjustment/triangulation
    print("Starting per chunk triangulation and bundle adjustment (if required)")
    n_processed = 0
    chunk_names = os.listdir(os.path.join(chunks_dir, "raw_chunks"))
    for chunk_name in chunk_names:
        in_dir = os.path.join(chunks_dir, "raw_chunks", chunk_name)
        out_dir = os.path.join(chunks_dir, "chunks", chunk_name)

        try:
            if len(submitted_jobs) >= args.n_jobs:
                submitted_jobs.pop(0).communicate()
            intermediate_dir = os.path.join(in_dir, "bundle_adjustment")
            if os.path.exists(intermediate_dir):
                print(f"{intermediate_dir} exists! Per chunk triangulation might crash!")
            if len(chunk_names) > 1:
                prepare_chunk_args = [
                        "python", f"preprocess/prepare_chunk.py",
                        "--raw_chunk", in_dir, "--out_chunk", out_dir, 
                        "--images_dir", images_dir
                ]
                if args.skip_bundle_adjustment:
                    prepare_chunk_args.append("--skip_bundle_adjustment")
                job = subprocess.Popen(
                    prepare_chunk_args,
                    stderr=open(f"{in_dir}/log.err", 'w'), 
                    stdout=open(f"{in_dir}/log.out", 'w'),
                )
                submitted_jobs.append(job)
                n_processed += 1
                print(f"Launched triangulation for [{n_processed} / {len(chunk_names)} chunks].")
                print(f"Logs in {in_dir}/log.err (or .out)")
            else:
                # if only one chunk, just copy from global for performance
                os.makedirs(out_dir, exist_ok=True)
                shutil.copyfile(f"{in_dir}/center.txt", f"{out_dir}/center.txt")
                shutil.copyfile(f"{in_dir}/extent.txt", f"{out_dir}/extent.txt")
                os.makedirs(f"{out_dir}/sparse/0", exist_ok=True)
                shutil.copyfile(f"{colmap_dir}/sparse/0/cameras.bin", f"{out_dir}/sparse/0/cameras.bin")
                shutil.copyfile(f"{colmap_dir}/sparse/0/images.bin", f"{out_dir}/sparse/0/images.bin")
                shutil.copyfile(f"{colmap_dir}/sparse/0/points3D.bin", f"{out_dir}/sparse/0/points3D.bin")
                if args.with_mvs:
                    shutil.copyfile(f"{colmap_dir}/sparse/0/cameras.bin", f"{out_dir}/sparse/cameras.bin")
                    shutil.copyfile(f"{colmap_dir}/sparse/0/images.bin", f"{out_dir}/sparse/images.bin")
                    shutil.copyfile(f"{colmap_dir}/sparse/0/points3D.bin", f"{out_dir}/sparse/points3D.bin")
                    convert_to_mvs_args = [
                            "InterfaceCOLMAP", 
                            "-w", f"{out_dir}",
                            "-i", ".",
                            "-o", "./openmvs_scene.mvs",
                            "--image-folder", "../../rectified/images",
                        ]
                    try:
                        subprocess.run(convert_to_mvs_args, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Error executing InterfaceCOLMAP: {e}")
                    densify_pointcloud_args = [
                            "DensifyPointCloud", "./openmvs_scene.mvs",
                            "-w", f"{out_dir}",
                            "--remove-dmaps", "1",
                            "-o", "./points3D.ply"
                        ]
                    try:
                        subprocess.run(densify_pointcloud_args, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Error executing DensifyPointCloud: {e}")
                    shutil.copyfile(f"{out_dir}/points3D.ply", f"{out_dir}/sparse/0/points3D.ply")
        except subprocess.CalledProcessError as e:
            print(f"Error executing prepare_chunk.py: {e}")
            sys.exit(1)


    for job in submitted_jobs:
        job.communicate()

    # create chunks.txt file that concatenates all chunks center.txt and extent.txt files
    try:
        subprocess.run([
            "python", "preprocess/concat_chunks_info.py",
            "--base_dir", os.path.join(chunks_dir, "chunks"),
            "--dest_dir", colmap_dir
        ], check=True)
        n_processed += 1
    except subprocess.CalledProcessError as e:
        print(f"Error executing concat_chunks_info.sh: {e}")
        sys.exit(1)

    # remove non-needed raw_chunks if needed.
    if not args.keep_raw_chunks:
        shutil.rmtree(os.path.join(chunks_dir, "raw_chunks"))
        print(f'raw_chunks folder has been deleted.')

    end_time = time.time()
    print(f"chunks successfully prepared in {(end_time - start_time)/60.0} minutes.")

