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
    
    gt = env.robots[0].get_position()
    gt_f = open(f"{pth}/gt/{index}.txt", "w", encoding="utf-8")
    gt_f.write(f"{gt[0]},{gt[1]},{gt[2]}\n")
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