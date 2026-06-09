import argparse
import os
import numpy as np
import cv2

from datetime import datetime
import time

###############################################
# Resolution of the cartesian form of the radar scan in metres per pixel
cart_resolution = .125  # m
# Cartesian visualisation size (used for both height and width)
cart_pixel_width = 2001  # pixels
# data path
path = "/root/oxford_radar_robotcar_dataset_sample_large/"
# output folder name
directory = "radar_cart_0.125_2001_timestamped1"
###############################################

def UnixTimeToSec(unix_timestamp):
    time_obj = datetime.fromtimestamp(unix_timestamp / 1_000_000)
    s = unix_timestamp % 1_000_000
    sec_timestamp = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + (float(s) / 1_000_000)
    return sec_timestamp

def main():
    print(" Starting radar image conversion...")

    dir_path = os.path.join(path, directory)
    os.makedirs(dir_path, exist_ok=True)
    print(f" Output directory prepared: '{directory}'")

    for foldername in os.listdir(path):
        folder_path = os.path.join(path, foldername)
        if not os.path.isdir(folder_path):
            continue

        print(f"\n Processing folder: {foldername}")

        timestamps_path = os.path.join(folder_path, "radar.timestamps")
        if not os.path.isfile(timestamps_path):
            print(" Skipping: No 'radar.timestamps' file found.")
            continue

        output_folder = os.path.join(dir_path, foldername)
        os.makedirs(output_folder, exist_ok=True)
        print(f" Output subdirectory created: {output_folder}")

        interpolate_crossover = True
        radar_timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)

        print(f" Number of radar timestamps: {len(radar_timestamps)}")
        print(f" First timestamp: {radar_timestamps[0]}, Last: {radar_timestamps[-1]}")

        radar_init_t = UnixTimeToSec(radar_timestamps[0])
        radar_t_list = np.array([], dtype=str)

        converted_count = 0

        for radar_timestamp in radar_timestamps:
            if converted_count >= 2:
                print(" Finished processing first 2 radar images.\n")
                break

            print(f"\n Processing radar timestamp: {radar_timestamp}")
            curr_radar_t = UnixTimeToSec(radar_timestamp) - radar_init_t
            curr_radar_t_str = str("%010.5f" % curr_radar_t)
            radar_t_list = np.append(radar_t_list, [curr_radar_t_str], axis=0)

            filename = os.path.join(folder_path, "radar", f"{radar_timestamp}.png")
            if not os.path.isfile(filename):
                print(f" Radar image missing: {filename}")
                continue

            print(f" Loading radar image: {filename}")
            radar_resolution = np.array([0.0432], np.float32)
            encoder_size = 5600

            raw_example_data = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            print(f" Raw image shape: {raw_example_data.shape}")

            timestamps = raw_example_data[:, :8].copy().view(np.int64)
            azimuths = (raw_example_data[:, 8:10].copy().view(np.uint16) / float(encoder_size) * 2 * np.pi).astype(np.float32)
            print(f" Azimuths shape: {azimuths.shape}\n First 10: {azimuths[:10].flatten()}")

            fft_data = raw_example_data[:, 11:].astype(np.float32)[:, :, np.newaxis] / 255.
            print(f" FFT data shape: {fft_data.shape}, dtype: {fft_data.dtype}")
            print(" FFT data sample [0:5, 0:5]:")
            print(fft_data[0:5, 0:5, 0])

            if (cart_pixel_width % 2) == 0:
                cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
            else:
                cart_min_range = cart_pixel_width // 2 * cart_resolution

            print(f" Cartesian grid min/max range: ±{cart_min_range:.2f} meters")

            coords = np.linspace(-cart_min_range, cart_min_range, cart_pixel_width, dtype=np.float32)
            Y, X = np.meshgrid(coords, -coords)
            print(f" Grid shape (X, Y): {X.shape}, dtype: {X.dtype}")
            print(" Sample grid X[1000:1005, 1000:1005]:")
            print(X[1000:1005, 1000:1005])
            print(" Sample grid Y[1000:1005, 1000:1005]:")
            print(Y[1000:1005, 1000:1005])

            sample_range = np.sqrt(Y * Y + X * X)
            sample_angle = np.arctan2(Y, X)
            sample_angle += (sample_angle < 0).astype(np.float32) * 2. * np.pi

            azimuth_step = azimuths[1] - azimuths[0]
            print(f" Azimuth step (radians): {azimuth_step.item():.6f}")

            sample_u = (sample_range - radar_resolution / 2) / radar_resolution
            sample_v = (sample_angle - azimuths[0]) / azimuth_step
            sample_u[sample_u < 0] = 0

            if interpolate_crossover:
                fft_data = np.concatenate((fft_data[-1:], fft_data, fft_data[:1]), 0)
                sample_v += 1
                print(" Interpolated crossover applied")

            polar_to_cart_warp = np.stack((sample_u, sample_v), -1)
            cart_img = np.expand_dims(cv2.remap(fft_data, polar_to_cart_warp, None, cv2.INTER_LINEAR), -1)
            cart_img = cart_img * 255.0
            print(" Sample warp coords [1000:1005, 1000:1005]:")
            print(polar_to_cart_warp[1000:1005, 1000:1005])

            print(f" Cartesian image shape: {cart_img.shape}, min: {cart_img.min()}, max: {cart_img.max()}")
            print(" Cartesian image values [1000:1005, 1000:1005]:")
            print(cart_img[1000:1005, 1000:1005, 0])


            out_img_path = os.path.join(output_folder, f"{curr_radar_t_str}.png")
            out_txt_path = os.path.join(output_folder, 'radar_t_list')

            print(f" Saving image to: {out_img_path}")
            cv2.imwrite(out_img_path, cart_img.astype(np.uint8))
            np.savetxt(out_txt_path, radar_t_list, delimiter=' ', fmt='%s')
            print("Saved radar_t_list")

            converted_count += 1

    print("\n All radar image conversions completed. Exiting...")

if __name__ == '__main__':
    main()
