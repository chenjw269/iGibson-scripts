import json


f = open('../scene.json', "r")
scene_json = json.load(f)

scene_list = list(scene_json.keys())
scene_dict = scene_json


import os
import cv2
import numpy as np


def sample_data(env, index, pth):
    
    rgb, depth, seg, ins_seg, normal = env.simulator.renderer.render_robot_cameras(modes=("rgb", "3d", "seg","ins_seg", "normal"))
    
    # RGB
    if not os.path.exists(f"{pth}/rgb/"):
        os.makedirs(f"{pth}/rgb/")
    
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    rgb = rgb * 255
    cv2.imwrite(f"{pth}/rgb/{index}.jpg", rgb)
    
    # # 3d
    # np.save(f"{pth}/3d/{index}.npy", depth[:, :, :3])
    
    # depth
    if not os.path.exists(f"{pth}/d/"):
        os.makedirs(f"{pth}/d/")
    
    depth_cp = np.linalg.norm(depth[:, :, :3], axis=2)
    depth_cp /= (depth_cp.max() + 1e-5)
    depth[:, :, :3] = depth_cp[..., None]
    depth = depth * 255
    cv2.imwrite(f"{pth}/d/{index}.jpg", depth)
     
    # semantic segmentation
    if not os.path.exists(f"{pth}/seg/"):
        os.makedirs(f"{pth}/seg/")
    
    MAX_CLASS_COUNT = 512
    seg = (seg[:, :, 0:1] * MAX_CLASS_COUNT).astype(np.int32)
    # colors = matplotlib.cm.get_cmap("plasma", 16)
    # seg_img = np.squeeze(colors(seg), axis=2) * 255
    # cv2.imwrite(f"{pth}/seg/{index}.jpg", seg_img)
    np.save(f"{pth}/seg/{index}.npy", seg)
    
    # # instance segmentation
    # MAX_INSTANCE_COUNT = 1024
    # ins_seg = (ins_seg[:, :, 0:1] * MAX_INSTANCE_COUNT).astype(np.int32)
    # np.save(f"{pth}/ins_seg/{index}.npy", ins_seg)
    
    # # intrinsic matrix
    # intrinsic = env.simulator.renderer.get_intrinsics()
    # np.save(f"{pth}/ins/{index}.npy", intrinsic)
    
    # # extrinsic matrix
    # extrinsic = env.simulator.renderer.V
    # np.save(f"{pth}/proj/{index}.npy", extrinsic)
    
    # # normal
    # np.save(f"{pth}/normal/{index}.npy", normal)
    
    # ground truth
    if not os.path.exists(f"{pth}/gt/"):
        os.makedirs(f"{pth}/gt/")
    
    # gt = env.robots[0].get_position()
    # gt_f = open(f"{pth}/gt/{index}.txt", "w", encoding="utf-8")
    # gt_f.write(f"{gt[0]},{gt[1]},{gt[2]},")
    # gt = env.robots[0].get_orientation()
    # gt_f.write(f"{gt[0]},{gt[1]},{gt[2]},{gt[3]}\n")
    # gt_f.close()    


def euler_to_quaternion(roll_deg, pitch_deg, yaw_deg):
    """
    Convert Euler angles (roll, pitch, yaw) in degrees to a quaternion.
    
    :param roll_deg: Rotation around the X-axis (in degrees)
    :param pitch_deg: Rotation around the Y-axis (in degrees)
    :param yaw_deg: Rotation around the Z-axis (in degrees)
    :return: Quaternion (w, x, y, z)
    """
    roll = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = round(cy * cp * cr + sy * sp * sr, 8)
    x = round(cy * cp * sr - sy * sp * cr, 8)
    y = round(sy * cp * sr + cy * sp * cr, 8)
    z = round(sy * cp * cr - cy * sp * sr, 8)

    return [x, y, z, w]


import os
import yaml
import numpy as np
import random

from tqdm import tqdm


import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.render.profiler import Profiler


def sample_scene(s_idx, r_idx, n_sample=None):
    
    scene_id = scene_list[s_idx]
    
    # Read sample positions
    pos_pth = f"E:/Workspace/Datasets/iGibson-dataset/iGibson-panorama-obj/"
    pos_pth = f"{pos_pth}/{scene_id}/{r_idx}/{scene_id}.txt"
    
    poses = []
    pos_fp = open(pos_pth, "r", encoding="utf-8")
    for pos_l in pos_fp:
        pos_l = pos_l.replace("\n", "")
        pos_l = pos_l.split(",")
        poses.append((float(pos_l[0]), float(pos_l[1]), 1.0, float(pos_l[2])))
    
    if n_sample is not None:
        assert n_sample < len(poses)
    else:
        n_sample = len(poses)

    # Sample data
    tgt_ds_pth = "E:/Workspace/Datasets/iGibson-dataset/iGibson-pano-obj-data/"
    tgt_pth = f"{tgt_ds_pth}/{scene_id}/{r_idx}/"
        
    config_filename = os.path.join(igibson.configs_path, "turtlebot_nav.yaml")
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    
    # 启动阴影和基于物理的渲染 (optional)
    config_data["enable_shadow"] = True
    config_data["enable_pbr"] = True
    config_data["output"] = ["rgb", "depth", "seg", "ins_seg", "normal"]
    
    # config_data["load_object_categories"] = []  # Uncomment this line to accelerate loading with only the building
    config_data["visible_target"] = False
    config_data["visible_path"] = False
    # Set specific scene
    config_data["scene_id"] = scene_id
    
    env = iGibsonEnv(config_file=config_data, scene_id=scene_id, mode="gui_interactive")
    s = env.simulator

    # Set a better viewing direction
    # s.viewer.initial_pos = [-2, 1.4, 1.2]
    # s.viewer.initial_view_direction = [0.6, -0.8, 0.1]
    # s.viewer.reset_viewer()
    
    print(f"Resetting environment... scene {scene_id}, room {r_idx}")
    env.reset()
        
    # Sample data of each position
    for p_idx in tqdm(range(len(poses)), desc=f"{scene_id} {r_idx}"):
        
        env.reset()
        # print("==============================")
        # with Profiler("Environment action step"):
        state, reward, done, info = env.step([0.0, 0.0])
        
        position = poses[p_idx][:3]
        env.robots[0].set_position(position)
        
        # rotation_angle = random.uniform(0, 360)
        # rotation = euler_to_quaternion(0.0, 0.0, rotation_angle)
        # rotation = poses[p_idx][3]
        # rotation = euler_to_quaternion(0.0, 0.0, rotation)
        # env.robots[0].set_orientation(rotation)
        
        index = "{:04d}".format(p_idx)
        # sample_data(env, index, tgt_pth)
        
        if not os.path.exists(f"{tgt_pth}/{p_idx}/gt/"):
            os.makedirs(f"{tgt_pth}/{p_idx}/gt/")
        gt_f = open(f"{tgt_pth}/{p_idx}/gt/0000.txt", "w", encoding="utf-8")
        gt_f.write(f"{position[0]},{position[1]},{position[2]}")
        
        # rotate and sample data
        for idx, deg in enumerate(range(0, 360, 45)):
            q = euler_to_quaternion(0.0, 0.0, deg)
            env.robots[0].set_orientation(q)
            index = "{:04d}".format(0) + f"_{idx}"
            # 
            sample_data(env, index, f"{tgt_pth}/{p_idx}")
            
            # robot_position = env.robots[0].get_position()
            # print(robot_position)
            # robot_orientation = env.robots[0].get_orientation()
            # print(robot_orientation)


import multiprocessing


def main():
    
    s_idxs = [1, 3, 4, 5, 6]
    r_nums = [5, 2, 6, 3, 6]
    
    processes = []

    for i in range(len(s_idxs)):
        
        s_idx = s_idxs[i]
        
        for r_idx in range(r_nums[i]):
            
            p = multiprocessing.Process(target=sample_scene, args=(s_idx, r_idx))
            processes.append(p)
            p.start()
            
    for process in processes:
        process.join()
        

if __name__ == "__main__":

    main()