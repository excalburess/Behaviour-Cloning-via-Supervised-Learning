import numpy as np
import time
import os
import pandas as pd

# Headless mode and step cap for CI / direct runs
headless = os.environ.get("ROBOENV_HEADLESS", "").strip().lower() in ("1", "true", "yes")
max_steps = int(os.environ.get("ROBOENV_MAX_STEPS", "40000"))

import matplotlib
if headless:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pybullet as p
import pinocchio as pin
from simulation_and_control import pb, MotorCommands, PinWrapper
from robotnik_arm_control import (
    ArmControllerParams, BaseControllerParams, BaseNavState, StopCriteria,
    arm_controller, base_controller, check_stop, configure_gripper_fingers, quaternion2bearing,
)


# ==============================================================================
# TARGET SPECIFICATION
# Fixed 3D Cartesian coordinates for the Kinova end-effector in the world frame:
# [X (forward/meters), Y (lateral/meters), Z (height/meters)]
TARGET_EE_POS = np.asarray([0.0, 1.1, 0.5])

# Fixed gripper finger configuration (angles in radians, held fixed during simulation):
# [finger_1, finger_tip_1, finger_2, finger_tip_2, finger_3, finger_tip_3]
# Set to None to use initial configuration from config/URDF, or an array of 6 angles.
FIXED_FINGER_ANGLES = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

# Max time (s) allowed for the base to reach its standoff pose before the arm starts reaching.
# Targets behind the robot need a large turn, so this must be generous.
BASE_NAV_TIMEOUT = 25.0
# Max EE distance from the shoulder the arm is asked to reach (m). Targets further away are projected
# onto this sphere, so the arm settles at the closest pose instead of fully stretching (singularity).
ARM_REACH_LIMIT = 0.85
# Base-to-target horizontal standoff (m) as a function of target height: larger for low targets,
# default for mid heights, smaller for high targets (far above the shoulder, which sits ~0.49 m high).
STANDOFF_Z_POINTS = [0.1, 0.5, 1.1, 1.25]
STANDOFF_POINTS = [0.80, 0.65, 0.65, 0.35]
# Stop once the EE is this close to the target (m); max iterations set via ROBOENV_MAX_STEPS
REACH_TOLERANCE = 0.01

#data collection constants
NOISE_HOLD_STEPS = 150 #redraws noise per 150 steps 
WHEEL_NOISE_SCALE = 2.0 #tunable
TORQUE_NOISE_SCALE = np.full(7, 5.0) #per all 7 joints scale by 5newtons
TORQUE_LIMIT = 30.0
PIN_THRESHOLD = 0.02
# ==============================================================================

# train_ranges = {
#     "distance": [(-3, 3)],
#     "angle": [(-np.pi, np.pi)],
#     "height": [(0.15, 1.3)],
#     "start_xy": [(-1.5, 1.5)],
#     "arm_offset" : [()]
#     
#

# }

def sampling(rng):
    pass

#pass in values for changing: target, xy and arm yaw
def run_episode(target, start_xy, start_yaw, start_arm_offset, seed, noise_seed=None, noise_std=0.0,
                headless=True, floor_friction=1.0, max_steps=40000, joint_margin=0.05, verbose=False):
    
    target = np.asarray(target, dtype=float)
    rng = np.random.default_rng(seed if noise_seed is None else noise_seed) #only for noise 
    start_arm_offset = np.asarray(start_arm_offset, dtype=float)
    
    # --------------------------------------------------------------------------
    # Configuration and Simulation Setup
    # --------------------------------------------------------------------------
    conf_file_name = "robotnik_arm.json"
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=not headless)

    # Configure realistic ground contact and skid-steer friction dynamics
    sim.SetFloorFriction(floor_friction)
    client = sim.pybullet_client
    rid = sim.bot[0].bot_pybullet

    #wheel joint ids
    wheel_joint_ids = [
        j for j in range(client.getNumJoints(rid))
        if "wheel" in client.getJointInfo(rid, j)[1].decode()
    ]
    for j in wheel_joint_ids:
        client.changeDynamics(
            rid, j,
            contactStiffness=30000,
            contactDamping=1000,
            anisotropicFriction=[1.0, 0.01, 1.0]
        )
    sim.bot[0].servo_motor_model.set_motor_gains(400, 30)

    #arm joint ids:
    arm_joint_ids = []
    for j in range(client.getNumJoints(rid)):
        info = client.getJointInfo(rid, j)
        name = info[1].decode()
        if (name.startswith("robot_j2s7s300_joint_")
                and "finger" not in name
                and info[2] != p.JOINT_FIXED):
            arm_joint_ids.append(j)
    #check for arm joints correct amount:
    assert len(arm_joint_ids) == 7, arm_joint_ids

    


    # --------------------------------------------------------------------------
    # Gripper Fingers: Fix gripper finger configuration
    # --------------------------------------------------------------------------
    target_finger_angles = configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)

    # Visualize the target reach reference in the PyBullet 3D environment as a red dot
    target_pos_list = target.tolist()
    visual_shape_id = client.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.03,
        rgbaColor=[1.0, 0.0, 0.0, 0.85]
    )
    client.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=target_pos_list
    )
    client.addUserDebugPoints(
        pointPositions=[target_pos_list],
        pointColorsRGB=[[1.0, 0.0, 0.0]],
        pointSize=10
    )

    # Active joints from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names_arr = np.expand_dims(np.array(ext_names), axis=0)
    source_names = ["pybullet"]

    # Pinocchio dynamic & kinematic model
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names_arr, source_names, False, 0, cur_dir)

    controlled_frame_name = "robot_j2s7s300_end_effector"
    ee_frame_id = dyn_model.pin_model.getFrameId(controlled_frame_name)
    shoulder_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_link_2")

    #simulation setup
    #robot base
    default_position, default_orientation = client.getBasePositionAndOrientation(rid)
    roll, pitch, _ = p.getEulerFromQuaternion(default_orientation)
    start_quat = p.getQuaternionFromEuler([roll, pitch, start_yaw])
    client.resetBasePositionAndOrientation(rid, [start_xy[0], start_xy[1], default_position[2]], start_quat)

    #robot arm
    applied_offset = start_arm_offset.copy()
    for jid, offset in zip(arm_joint_ids, start_arm_offset):
        default_angle = client.getJointState(rid, jid)[0]
        client.resetJointState(rid, jid, targetValue = default_angle+offset, targetVelocity = 0.0)

    

    #refresh readings before init is used
    sim.ReceiveObservation()

    init_joint_angles = np.array(sim.GetMotorAngles(0))
    init_base_pos = np.array(sim.GetBasePosition()) 
    init_base_ori = np.array(sim.GetBaseOrientation())

    # Forward kinematics for initial end-effector position
    init_q_all = np.concatenate((init_base_pos, init_base_ori, init_joint_angles))
    init_q_pin = dyn_model.ReoderJoints2PinVec(init_q_all, "pos")
    pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, init_q_pin)
    pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
    init_ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()

    if verbose:
        print("==================================================================")
        print(f"Target Cartesian EE Position : {target}")
        print(f"Initial Cartesian EE Position: {np.round(init_ee_pos, 4)}")
        print(f"Initial Cartesian Error (m)  : {np.linalg.norm(target - init_ee_pos):.4f}")
        print("==================================================================")

    # Joint limits
    lower_limits, upper_limits = sim.GetBotJointsLimit()

    time_step = sim.GetTimeStep()
    current_time = 0.0

    # --------------------------------------------------------------------------
    # Coordinated Motion Planning: Base Navigation & Arm Reach
    # --------------------------------------------------------------------------
    # Standoff distance between robot base center and target, depending on target height.
    standoff = float(np.interp(target[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
    target_xy = target[:2] - init_base_pos[:2]
    target_bearing = np.arctan2(target_xy[1], target_xy[0]) if np.linalg.norm(target_xy) > 1e-4 else start_yaw

    # Mobile base goal location facing the target
    base_goal_pos = target[:2] - standoff * np.array([np.cos(target_bearing), np.sin(target_bearing)])
    base_goal_bearing = target_bearing

    # Base navigation controller (align -> drive -> orient -> done), see BaseControllerParams
    base_state = BaseNavState()
    base_navigation_done = base_state.done
    base_params = BaseControllerParams(nav_timeout=BASE_NAV_TIMEOUT)

    # Arm controller (differential IK + joint PD), see ArmControllerParams
    arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
    q_init_arm = init_joint_angles[4:11].copy()
    q_des_arm = q_init_arm.copy()
    qd_des_arm = np.zeros(7)
    arm_lower = np.array(lower_limits[4:11])
    arm_upper = np.array(upper_limits[4:11])
    revolute = arm_lower < arm_upper

    # Stop conditions (step cap, target reached, unreachable-target stall), see StopCriteria
    stop_criteria = StopCriteria(reach_tolerance=REACH_TOLERANCE, max_steps=max_steps)

    # Command interface
    cmd = MotorCommands()

    # Data logging for plotting
    time_all = []
    ee_pos_all = []
    ee_error_all = []
    base_pos_all = []
    q_mes_all = []
    qd_mes_all = []
    q_d_all = []
    qd_d_all = []

    #data logging for experiments
    data = []
    step = 0
    n_steps_pinned = 0
    nav_done_time = None
    stop_message = None
    noise = np.zeros(11)

    # --------------------------------------------------------------------------
    # Main Simulation Loop
    # --------------------------------------------------------------------------
    while True:
        # Measure current states
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        q_mes = np.array(sim.GetMotorAngles(0))
        qd_mes = np.array(sim.GetMotorVelocities(0))

        #extra sensor data
        base_lin_vel, base_ang_vel = client.getBaseVelocity(rid)
       
        # Full kinematic state vector in Pinocchio format
        q_all = np.concatenate((base_pos, base_ori, q_mes))
        q_pin = dyn_model.ReoderJoints2PinVec(q_all, "pos")

        # Compute end-effector forward kinematics
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        ee_error = target - ee_pos
        ee_dist_error = np.linalg.norm(ee_error)

        shoulder_pos = dyn_model.pin_data.oMf[shoulder_frame_id].translation.copy()
        q_arm = q_mes[4:11]
        lim_margin = np.where(revolute, np.minimum(q_arm - arm_lower, arm_upper - q_arm), np.nan)
        if np.any(lim_margin[revolute] < PIN_THRESHOLD):
            n_steps_pinned += 1

        # ----------------------------------------------------------------------
        # 1. Mobile Base Control (Wheel velocities)
        # ----------------------------------------------------------------------
        wheel_cmds = base_controller(
            base_state, base_pos, base_bearing, base_goal_pos, base_goal_bearing, current_time, base_params
        )
        base_navigation_done = base_state.done

        # ----------------------------------------------------------------------
        # 2. Arm Control (reaching via differential IK + gravity-compensated joint PD)
        # ----------------------------------------------------------------------
        # While the base is travelling the arm holds its nominal configuration.
        torque_arm, q_des_arm, qd_des_arm = arm_controller(
            dyn_model, q_pin, q_mes[4:11], qd_mes[4:11], q_des_arm, q_init_arm, target, ee_pos,
            ee_frame_id, shoulder_frame_id, arm_lower, arm_upper, time_step, arm_params,
            reach_active=base_navigation_done,
        )

        #tau_g computation + navigation done field.
        tau_g = pin.computeGeneralizedGravity(dyn_model.pin_model, dyn_model.pin_data, q_pin)[6:13]
    
        if base_navigation_done and nav_done_time is None:
            nav_done_time = current_time

        #noise:

        if noise_std > 0 and step % NOISE_HOLD_STEPS == 0:
            n_r, n_l = (0.0, 0.0) if base_navigation_done else rng.normal(0, noise_std * WHEEL_NOISE_SCALE, 2) #if not moving dont apply noise if so do
            n_tau = rng.normal(0, 1, 7) * noise_std * TORQUE_NOISE_SCALE #gives 7 random numbers +-1 per arm joint (loc(centre of cluster), scale(standard dev), size(how many nums))
            noise = np.concatenate(([n_r, n_l, n_l, n_r], n_tau)) 
        if base_navigation_done:
            noise[:4] = 0.0 #switch off noise once parked

        # ----------------------------------------------------------------------
        # 3. Gripper / Finger Control (Fingers locked at fixed configuration)
        # ----------------------------------------------------------------------
        torque_fingers = np.zeros(6)
       
        # ----------------------------------------------------------------------
        # Send Multi-Interface Control Commands
        # ----------------------------------------------------------------------
        # 4 wheels (velocity) + 7 arm joints (torque) + 6 fingers (torque)
        # cmd_all = np.concatenate((wheel_cmds, torque_arm, torque_fingers)) this is now passed in cmd_all below

        #commands with noise sent to expert
        clean_cmds = np.concatenate((wheel_cmds, torque_arm))
        sent_cmd = clean_cmds + noise
        sent_cmd[4:] = np.clip(sent_cmd[4:], -TORQUE_LIMIT, TORQUE_LIMIT) #limits joint values to between -torque limit and +torque limit
        cmd_all = np.concatenate((sent_cmd, torque_fingers))

       
        control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
        cmd.SetControlCmd(cmd_all, control_modes)
        sim.Step(cmd, "torque")

        # Visualization in Pinocchio GUI if active
        if dyn_model.visualizer:
            for index in range(len(sim.bot)):
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)

        # Exit logic with 'q' key in PyBullet GUI window
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        # Stop criteria: maximum steps, target reached within fine tolerance, or unreachable target
        stop_message = check_stop(stop_criteria, current_time, time_step, ee_dist_error, base_navigation_done)
        if stop_message:
            print(stop_message)
            break

        # Data logging for plotting
        time_all.append(current_time)
        ee_pos_all.append(ee_pos)
        ee_error_all.append(ee_error)
        base_pos_all.append(base_pos)
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        current_desired_q = np.concatenate((q_mes[:4], q_des_arm, target_finger_angles))
        current_desired_qd = np.concatenate((wheel_cmds, qd_des_arm, np.zeros(6)))
        q_d_all.append(current_desired_q)
        qd_d_all.append(current_desired_qd)

        data.append({
            "step": step, "t": current_time,
            #base state
            "base_x": base_pos[0], "base_y" : base_pos[1], "base_z" : base_pos[2],
            "base_yaw" : base_bearing,
            "base_qx": base_ori[0], "base_qy": base_ori[1], "base_qz": base_ori[2], "base_qw": base_ori[3],
            "base_vx" : base_lin_vel[0],  "base_vy" : base_lin_vel[1],  "base_vz" : base_lin_vel[2],
            "base_wx" : base_ang_vel[0],   "base_wy" : base_ang_vel[1],  "base_wz" : base_ang_vel[2],

            #wheels, arm, fingers
            **{f"wheel_angle{i}": q_mes[i] for i in range(4)},
            **{f"wheel_velocity{i}" : qd_mes[i] for i in range(4)},
            **{f"q{i}": q_mes[4+i] for i in range(7)}, #joint angle
            **{f"qd{i}": qd_mes[4+i] for i in range(7)}, #joint velocity
            **{f"finger{i}": q_mes[11 + i] for i in range(6)},

            #end-effector
            "ee_x": ee_pos[0], "ee_y": ee_pos[1], "ee_z": ee_pos[2],

            #expert added data
            "phase": base_state.phase, "nav_done": base_navigation_done,  #internal memory
            **{f"q_des{i}": q_des_arm[i] for i in range(7)}, #arm desired angle for each joint
            **{f"qd_des{i}": qd_des_arm[i] for i in range(7)}, #arm desired velocity for each joint
            **{f"tau_g{i}": tau_g[i] for i in range(7)}, #tau_g (extra data) gravity torque (predict torque - gravity)

            #controller data being sent 
            **{f"cmd_wheel{i}": wheel_cmds[i] for i in range(4)},
            **{f"torque_cmd{i}": torque_arm[i] for i in range(7)},

            **{f"lim_margin{i}": lim_margin[i] for i in range(7)},                                                          #Distance from each joint to its nearest limit
            "shoulder_x": shoulder_pos[0], "shoulder_y": shoulder_pos[1], "shoulder_z": shoulder_pos[2],                    #Expert reaches relative to shoulder
            "noise_wheel_r": noise[0], "noise_wheel_l": noise[1],                                                           #records what noise was applied (analysis)
            **{f"noise_tau{i}": noise[4 + i] for i in range(7)},                                                                          
        })

        step+=1

        if not headless:
            time.sleep(0.001)

        current_time  = step * time_step 
        if int(current_time / time_step) % 500 == 0:
            print(f"t={current_time:.2f}s | Base pos: {np.round(base_pos[:2], 3)} | EE pos: {np.round(ee_pos, 3)} | EE error: {ee_dist_error:.4f}m | yaw {base_bearing:.2f} phase {base_state.phase} nav_done {base_navigation_done}")



    df = pd.DataFrame(data)
    df["seed"] = seed
    df["target_x"], df["target_y"], df["target_z"] = target


    #per-episode summary report
    if stop_message is None:
        stop_reason = "quit"
    elif "step cap" in stop_message.lower():
        stop_reason = "step_cap"
    elif "not reach" in stop_message.lower():
        stop_reason = "unreachable"
    elif "reached" in stop_message.lower():
        stop_reason = "reached"
    else:
        stop_reason = "other"


    final_err = float(np.linalg.norm(target - ee_pos))
    best_ee_error = float(min(np.linalg.norm(df[["ee_x","ee_y","ee_z"]].to_numpy() - target, axis=1).min(), final_err)) if len(df) else final_err


    summary = {
        "seed": seed,
        "target_x": target[0], "target_y": target[1], "target_z": target[2],
        "start_x": start_xy[0], "start_y": start_xy[1], "start_yaw": start_yaw,
        **{f"arm_offset{i}": start_arm_offset[i] for i in range(7)},
        "noise_std": noise_std,
        "base_goal_x": base_goal_pos[0], "base_goal_y": base_goal_pos[1],
        "standoff": standoff,
        "stop_reason": stop_reason, "stop_message": stop_message,
        "final_ee_error": final_err,
        "best_ee_error": best_ee_error,
        "nav_done_time": nav_done_time,
        "nav_timed_out": nav_done_time is not None and nav_done_time >= BASE_NAV_TIMEOUT,
        "n_steps": step, "sim_duration": current_time,
        "max_finger_dev": float(np.max(np.abs(q_mes[11:17] - target_finger_angles))),
        **{f"applied_offset{i}": applied_offset[i] for i in range(7)},
        "floor_friction": floor_friction,
        "n_rows": len(df),
        "frac_steps_pinned": n_steps_pinned / max(step, 1),
        "final_phase": base_state.phase,
    }

    # # Final summary report
    # print("==================================================================")
    # print(f"Final Base Position          : {np.round(base_pos[:2], 4)}")
    # print(f"Final End-Effector Position  : {np.round(ee_pos, 4)}")
    # print(f"Target End-Effector Position : {target}")
    # print(f"Final Cartesian Error (m)    : {np.linalg.norm(target - ee_pos):.4f}")
    # print(f"Fixed Finger Angles          : {np.round(target_finger_angles, 4)}")
    # print(f"Final Finger Angles          : {np.round(q_mes[11:17], 4)}")
    # print(f"Max Finger Deviation (rad)   : {np.max(np.abs(q_mes[11:17] - target_finger_angles)):.6f}")
    # print("==================================================================")

    if headless:
        print(f"Headless run finished at t={current_time:.2f}s")
        client.disconnect()
        return df, summary

    # --------------------------------------------------------------------------
    # Plotting Performance
    # --------------------------------------------------------------------------
    ee_pos_all = np.array(ee_pos_all)
    ee_error_all = np.array(ee_error_all)
    time_all = np.array(time_all)

    # 1. Cartesian end-effector coordinates tracking
    plt.figure(figsize=(10, 8))
    coords = ['X', 'Y', 'Z']
    for idx in range(3):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_all, ee_pos_all[:, idx], label=f'Actual EE {coords[idx]}', color='b')
        plt.axhline(y=target[idx], color='r', linestyle='--', label=f'Target EE {coords[idx]}')
        plt.ylabel(f'{coords[idx]} Position (m)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.xlabel('Time (s)')
    plt.suptitle('End-Effector Cartesian Position Reaching')
    plt.tight_layout()
    plt.show()

    # 2. Cartesian distance error over time
    plt.figure(figsize=(8, 4))
    ee_dist_norm = np.linalg.norm(ee_error_all, axis=1)
    plt.plot(time_all, ee_dist_norm, label='EE Euclidean Error (m)', color='g')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.title('Cartesian End-Effector Reaching Error')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. Arm joint position tracking
    plt.figure(figsize=(10, 8))
    for i in range(7):
        plt.subplot(4, 2, i + 1)
        plt.plot(time_all, [q[4 + i] for q in q_mes_all], label=f'Measured J{i+1}')
        plt.plot(time_all, [q[4 + i] for q in q_d_all], label=f'Desired J{i+1}', linestyle='--')
        plt.ylabel(f'J{i+1} (rad)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
    plt.xlabel('Time (s)')
    plt.suptitle('Arm Joint Position Profiles')
    plt.tight_layout()
    plt.show()


    client.disconnect()
    return df, summary


if __name__ == '__main__':
    df, summary = run_episode(target=np.array([-3.0, -3.0, 1.3]), start_xy=np.array([1.0, 0.0]),
     start_yaw=np.pi / 2, start_arm_offset=np.zeros(7), seed=0, noise_std=0.0, noise_seed=0, verbose=True, headless=True)
    print(df.shape)
    print(df.columns.tolist())
    print(df[["t", "phase", "nav_done", "cmd_wheel0", "torque_cmd0"]].iloc[::2000])
    print(summary)
  
