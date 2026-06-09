import os
import subprocess
import time
import argparse
import shutil

# ----- Update these paths -----
# For dataset_root, use the path where the Oxford dataset is located
# Make sure that the directory contains same dataset for which the result being tested
dataset_root = "/root/dataset/"
gt_output_dir = "/root/catkin_ws/src/poses"
launch_file = "pw_ndt_radar_scan_matching radar_odometry.launch"
# results_base_dir = "/root/catkin_ws/src/res/th40/"
results = "/root/catkin_ws/src/results/result/data/grid_9_6_4"
eval_dir = "/root/catkin_ws/src/kitti_eval/devkit_for_oxford/cpp"

# ----- Generate GT Files -----
def generate_gt():
    script_path = "/root/catkin_ws/src/script/oxford_gen_gt_file.py"
    cmd = ["python3", script_path]
    env = os.environ.copy()
    env["oxford_data_path"] = dataset_root
    env["gt_file_save_path"] = gt_output_dir
    subprocess.run(cmd, env=env, check=True)
    print("Ground truth generated.\n")

    # Create save directory used in the launch file
    save_dir = os.path.join(results_base_dir)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        print(f"Removed existing directory: {save_dir}")
    os.makedirs(save_dir)
    print(f"Created save directory: {save_dir}")

# ----- Run each mode -----
def run_mode(name, use_pw, use_imu, radar_path, gt_path, save_path, bag_name):
    if str(use_imu).lower() == "true":
        cmd = [
            "roslaunch",
            launch_file,
            f"use_pw:={str(use_pw).lower()}",
            f"use_imu:={str(use_imu).lower()}",
            f"directory:={radar_path}/radar/",
            f"gt_directory:={gt_path}",
            f"save_directory:={save_path}",
            f"imu_directory:={radar_path}/gps/ins.csv"
        ]
    else:
        cmd = [
            "roslaunch",
            launch_file,
            f"use_pw:={str(use_pw).lower()}",
            f"use_imu:={str(use_imu).lower()}",
            f"directory:={radar_path}/radar",
            f"gt_directory:={gt_path}",
            f"save_directory:={save_path}",
            f"bag_name:={bag_name}"
        ]
    process = subprocess.Popen(" ".join(cmd), shell=True)
    try:
        process.wait(timeout=500)
    except subprocess.TimeoutExpired:
        process.terminate()
        time.sleep(5)
        process.kill()

# ----- Run Evaluation -----
def run_evaluation():
    for date in os.listdir(dataset_root):
        if len(date) == 36:  # Only valid sequence folders
            gt_path = os.path.join(gt_output_dir, "2019-01-10-11-46-21-radar-oxford-10k_gt.txt")
            
        os.chdir(eval_dir)

    compile_cmd = ["g++", "-std=c++17" , "-O3", "-DNDEBUG", "-o", "evaluate_odometry", "evaluate_odometry.cpp", "matrix.cpp"]
    try:
        print("Compiling evaluation code...")
        subprocess.run(compile_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("Compilation failed:", e)
        return
    print(dataset_root)
    for date in os.listdir(dataset_root):
        print(date)
        if len(date) == 36:
            gt_path = os.path.join(gt_output_dir, f"{date}_gt.txt")
            result_path = os.path.join(results, f"{date}_res.txt")
            print(gt_path)
            print(result_path)

            if os.path.exists(gt_path) and os.path.exists(result_path):
                try:
                    print(f"Running evaluation for {date}...")
                    
                    subprocess.run(
                        ["./evaluate_odometry", "result_"+date],
                        check=True,
                        env={**os.environ, "GT_PATH": gt_path, "RESULT_PATH": result_path}
                    )
                except subprocess.CalledProcessError as e:
                    print(f"Evaluation failed for {date}:", e)
        
# ----- Main Pipeline -----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "pw_ndt_with_imu",
            "pw_ndt_without_imu",
            "std_ndt_with_imu",
            "std_ndt_without_imu",
            "all"
        ],
        default="all",
        help="Choose which mode to run"
    )
    args = parser.parse_args()
    selected_mode = args.mode

    # generate_gt()

    # all_modes = [
    #     {"name": "pw_ndt_with_imu", "use_pw": True, "use_imu": True},
    #     {"name": "pw_ndt_without_imu", "use_pw": True, "use_imu": False},
    #     {"name": "std_ndt_with_imu", "use_pw": False, "use_imu": True},
    #     {"name": "std_ndt_without_imu", "use_pw": False, "use_imu": False}
    # ]

    # modes_to_run = (
    #     all_modes if selected_mode == "all"
    #     else [m for m in all_modes if m["name"] == selected_mode]
    # )

    # for date in os.listdir(dataset_root):
    #     if len(date) == 36:  # Only valid sequence folders
    #         radar_path = os.path.join(dataset_root, date)
    #         gt_path = os.path.join(gt_output_dir, f"{date}_gt.txt")

    #         for m in modes_to_run:
    #             save_path = results_base_dir
    #             os.makedirs(save_path, exist_ok=True)
    #             run_mode(
    #                 name=m["name"],
    #                 use_pw=m["use_pw"],
    #                 use_imu=m["use_imu"],
    #                 radar_path=radar_path,
    #                 gt_path=gt_path,
    #                 save_path=save_path,
    #                 bag_name=date
    #             )

    run_evaluation()

if __name__ == "__main__":
    main()
