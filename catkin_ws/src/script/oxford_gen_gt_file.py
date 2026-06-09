import argparse
import pandas as pd
import time
import os
import numpy as np
import time
import math
import errno 

oxford_data_path = os.environ.get("oxford_data_path", "/root/catkin_ws/oxford_radar_robotcar_dataset_sample_large/")
gt_file_save_path = os.environ.get("gt_file_save_path", "/root/catkin_ws/src/poses")

def GetMatFromXYYaw(x,y,yaw):
    tf_mat = np.matrix([[math.cos(yaw), -math.sin(yaw), 0.0, x]
                       ,[math.sin(yaw), math.cos(yaw), 0.0, y]
                       ,[0.0, 0.0, 1.0, 0.0]
                       ,[0.0, 0.0, 0.0, 1.0]])
    return tf_mat

def main():
    os.makedirs(gt_file_save_path, exist_ok=True)
    for filename in os.listdir(oxford_data_path):
      if len(filename) == 36:# and filename == "2019-01-10-11-46-21-radar-oxford-10k":
        print(filename)
        out_path = os.path.join(gt_file_save_path, f"{filename}_gt.txt")
        try:
            f = open(out_path, "w")
        except OSError as e:
            if e.errno == errno.EROFS:          # read-only file system
                fallback_dir = "/catkin_ws/src/poses"
                os.makedirs(fallback_dir, exist_ok=True)
                out_path = os.path.join(fallback_dir, f"{filename}_gt.txt")
                print(f"[WARN] {gt_file_save_path} is read-only; "
                       f"writing to {out_path} instead.")
                f = open(out_path, "w")
            else:
                raise
        path = oxford_data_path + filename + '/'
        ###--- Write Ground Truth Radar Odometry ---###
        ro_file = path+'gt/radar_odometry.csv'
        ro_data = pd.read_csv(ro_file)
        ro_timestamps = ro_data.iloc[:,8].copy()
        ro_index = 0
        ro_tf_old = GetMatFromXYYaw(0,0,0)
        ro_tf_curr = GetMatFromXYYaw(0,0,0)

        R_x_180 = np.eye(4)
        R_x_180[1, 1] = -1; R_x_180[2, 2] = -1

        for ro_timestamp in ro_timestamps:
          x = ro_data.iloc[ro_index,2]
          y = ro_data.iloc[ro_index,3]
          yaw = ro_data.iloc[ro_index,7]
          ro_tf = GetMatFromXYYaw(x,y,yaw)
          ro_tf_curr = ro_tf_old * ro_tf
          # coordinate transform
          ro_tf_curr_ = R_x_180.dot(ro_tf_curr).dot(R_x_180)
          # write to file
          f.write(str(ro_tf_curr_[0,0])+' '+str(ro_tf_curr_[0,1])+' '+ str(ro_tf_curr_[0,2]) +' '+str(ro_tf_curr_[0,3])+' '+
          str(ro_tf_curr_[1,0])+' '+str(ro_tf_curr_[1,1])+' '+ str(ro_tf_curr_[1,2]) +' '+str(ro_tf_curr_[1,3])+' '+
          str(ro_tf_curr_[2,0])+' '+str(ro_tf_curr_[2,1])+' '+ str(ro_tf_curr_[2,2]) +' '+str(ro_tf_curr_[2,3])+'\n')
          # update
          ro_tf_old = ro_tf_curr
          ro_index = ro_index+1

        print("End Writing GT")

if __name__ == '__main__':
  main()
