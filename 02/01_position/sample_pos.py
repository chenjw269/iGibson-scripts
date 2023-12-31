import os
import yaml
import numpy as np
import random

from tqdm import tqdm

from utils import *

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.render.profiler import Profiler


def sample_position(scene_id, pos_pth, tgt_pth, with_obj=False):
        
    # Read sample positions    
    pos = []
    pos_fp = open(pos_pth, "r", encoding="utf-8")
    for pos_l in pos_fp:
        pos_l = pos_l.replace("\n", "")
        pos_l = pos_l.split(",")
        pos.append((float(pos_l[0]), float(pos_l[1]), 1.0, float(pos_l[2])))

    # Simulation configuration
    config_filename = os.path.join(igibson.configs_path, "turtlebot_nav.yaml")
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    
    # 启动阴影和基于物理的渲染 (optional)
    config_data["enable_shadow"] = True
    config_data["enable_pbr"] = True
    config_data["output"] = ["rgb", "depth", "seg", "ins_seg", "normal"]
    
    if with_obj is False:
        config_data["load_object_categories"] = []  # Uncomment this line to accelerate loading with only the building
    
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
    
    print(f"Resetting environment... scene {scene_id}")
    env.reset()
        
    # Sample data of each position
    for p_idx in tqdm(range(len(pos)), desc=f"{scene_id}"):
        
        env.reset()
        # print("==============================")
        # with Profiler("Environment action step"):
        state, reward, done, info = env.step([0.0, 0.0])
        
        position = pos[p_idx][:3]
        env.robots[0].set_position(position)
        
        # rotation_angle = random.uniform(0, 360)
        # rotation = euler_to_quaternion(0.0, 0.0, rotation_angle)
        # rotation = poses[p_idx][3]
        # rotation = euler_to_quaternion(0.0, 0.0, rotation)
        # env.robots[0].set_orientation(rotation)
        
        index = "{:04d}".format(p_idx)
        # sample_data(env, index, tgt_pth)
        
        # if not os.path.exists(f"{tgt_pth}/{p_idx}/gt/"):
        #     os.makedirs(f"{tgt_pth}/{p_idx}/gt/")
        # gt_f = open(f"{tgt_pth}/{p_idx}/gt/0000.txt", "w", encoding="utf-8")
        # gt_f.write(f"{position[0]},{position[1]},{position[2]}")
        
        # rotate and sample data
        for idx, deg in enumerate(range(0, 360, 45)):
            q = euler_to_quaternion(0.0, 0.0, deg)
            env.robots[0].set_orientation(q)
            index = "{:04d}".format(0) + f"_{idx}"

            sample_data(env, index, f"{tgt_pth}/{p_idx}")
            
            # robot_position = env.robots[0].get_position()
            # print(robot_position)
            # robot_orientation = env.robots[0].get_orientation()
            # print(robot_orientation)