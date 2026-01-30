import time
import pylibfranka
import argparse
import numpy as np
import pygame
from tqdm import trange

def rotation_matrix_to_axis_angle(R, eps=1e-6):
    """
    Convert a 3x3 rotation matrix to axis-angle using NumPy.

    Parameters
    ----------
    R : (3, 3) ndarray
        Rotation matrix.
    eps : float
        Tolerance for detecting special cases (0 and pi).

    Returns
    -------
    axis : (3,) ndarray
        Unit vector (ux, uy, uz) representing the rotation axis.
    angle : float
        Rotation angle in radians in [0, pi].
    """
    R = np.asarray(R, dtype=float)
    assert R.shape == (3, 3), "R must be a 3x3 matrix"

    R11, R12, R13 = R[0, 0], R[0, 1], R[0, 2]
    R21, R22, R23 = R[1, 0], R[1, 1], R[1, 2]
    R31, R32, R33 = R[2, 0], R[2, 1], R[2, 2]

    # 1. Angle from trace
    trace = R11 + R22 + R33
    c = (trace - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)
    angle = float(np.arccos(c))

    # 2. Angle ~ 0 (identity)
    if abs(angle) < eps:
        return np.array([1.0, 0.0, 0.0]), 0.0

    # 3. Angle ~ pi
    if abs(np.pi - angle) < eps:
        ux2 = max(0.0, (R11 + 1.0) / 2.0)
        uy2 = max(0.0, (R22 + 1.0) / 2.0)
        uz2 = max(0.0, (R33 + 1.0) / 2.0)

        if ux2 >= uy2 and ux2 >= uz2:
            ux = np.sqrt(ux2)
            if ux > eps:
                uy = (R12 + R21) / (4.0 * ux)
                uz = (R13 + R31) / (4.0 * ux)
            else:
                uy, uz = 0.0, 0.0
        elif uy2 >= ux2 and uy2 >= uz2:
            uy = np.sqrt(uy2)
            if uy > eps:
                ux = (R12 + R21) / (4.0 * uy)
                uz = (R23 + R32) / (4.0 * uy)
            else:
                ux, uz = 0.0, 0.0
        else:
            uz = np.sqrt(uz2)
            if uz > eps:
                ux = (R13 + R31) / (4.0 * uz)
                uy = (R23 + R32) / (4.0 * uz)
            else:
                ux, uy = 0.0, 0.0

        axis = np.array([ux, uy, uz], dtype=float)
        norm = np.linalg.norm(axis)

        if norm < eps:
            return np.array([1.0, 0.0, 0.0]), angle

        return axis / norm, angle

    # 4. General case: 0 < angle < pi
    denom = 2.0 * np.sin(angle)
    ux = (R32 - R23) / denom
    uy = (R13 - R31) / denom
    uz = (R21 - R12) / denom
    axis = np.array([ux, uy, uz], dtype=float)

    # Normalize for safety
    axis /= np.linalg.norm(axis)
    return axis, angle

class FrankaController:
    def __init__(self, ip_address):
        self.robot = pylibfranka.Robot(ip_address)
        self.robot.automatic_error_recovery()
        self.gripper = pylibfranka.Gripper(ip_address)
        # self.gripper.homing()
        self.gripper.move(0.0, 0.04)

        # close the gripper


        joint_stiffness = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
        joint_damping = [70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0]

        self.joint_stiffness = np.array(joint_stiffness)
        self.joint_damping = np.array(joint_damping)

        self.robot.set_collision_behavior(
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        )
        self.active_controller = self.robot.start_torque_control()
        self.model = self.robot.load_model()

        self.current_dq = np.zeros(7)
        R, t = self.ee_pose()
        self.cur_targ_pos = t
        self.dur = 0.0

        self.torque_diff_log = []
        self.last_torque = np.zeros(7)
    
    def ee_pose_by_state(self, state):
        transform = np.array(state.O_T_EE).reshape(4, 4).T
        R = transform[:3, :3].copy()
        t = transform[:3, 3].copy()
        R = R @ np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        t += np.array([-0.00112884, -0.00100483, 0.10032279])

        return R, t

    def ee_pose(self):
        state = self.robot.read_once()
        return self.ee_pose_by_state(state)

    def apply_torque(self, tau_d):

        tau_d -= self.last_torque

        max_dim_val = 0.2
        for i in range(len(tau_d)):
            if tau_d[i] > max_dim_val:
                tau_d[i] = max_dim_val
            elif tau_d[i] < -max_dim_val:
                tau_d[i] = -max_dim_val

        tau_d += self.last_torque

        torque_command = pylibfranka.Torques(tau_d.tolist())
        torque_command.motion_finished = False
        self.active_controller.writeOnce(torque_command)

        if self.last_torque is not None:
            diff = np.array(tau_d) - np.array(self.last_torque)
            diff = np.max(np.abs(diff))
            # if diff > 0.1:
            #     print(f"Torque diff norm: {diff}")
        self.last_torque = tau_d

    def apply_targ_pos(self, targ_pos = None):
        if targ_pos is not None:
            self.cur_targ_pos = targ_pos
        
        state, dur = self.active_controller.readOnce()
        R, t = self.ee_pose_by_state(state)
        axis, angle = rotation_matrix_to_axis_angle(R)
        vel_full = np.zeros(6)
        vel_full[:3] = (self.cur_targ_pos - t) * 10.0
        vel_full[3:] = -axis * angle # * 5.0
        # make vel_full[3:] have max norm of 0.1
        max_rot_norm = 0.2
        norm_rot = np.linalg.norm(vel_full[3:])
        if norm_rot > max_rot_norm:
            vel_full[3:] = vel_full[3:] / norm_rot * max_rot_norm

        J = np.array(self.model.zero_jacobian(state)).reshape(7, 6).T
        # pseudo-inverse of J
        J_pinv = np.linalg.pinv(J)
        dq_desired = J_pinv @ vel_full

        for i in range(7):
            if dq_desired[i] > 0.5:
                dq_desired[i] = 0.5
            elif dq_desired[i] < -0.5:
                dq_desired[i] = -0.5

        state, dur = self.active_controller.readOnce()

        dq = np.array(state.dq)

        coriolis = np.array(self.model.coriolis(state))

        tau_task = self.joint_stiffness * dq_desired - self.joint_damping * dq
        tau_d = tau_task + coriolis

        self.apply_torque(tau_d)
    
    def apply_q(self, q_targ, duration=3.0):
        state, dur = self.active_controller.readOnce()
        start_q = np.array(state.q)

        t = 0.0
        joint_stiffness = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
        joint_damping = np.array([14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0])
        while t < duration:
            state, dur = self.active_controller.readOnce()
            t += dur.to_sec()
            alpha = min(t / duration, 1.0)
            q_desired = (1 - alpha) * start_q + alpha * q_targ

            q = np.array(state.q)
            dq = np.array(state.dq)

            coriolis = np.array(self.model.coriolis(state))

            position_error = q_desired - q
            tau_task = joint_stiffness * position_error - joint_damping * dq

            tau_d = tau_task + coriolis

            self.apply_torque(tau_d)
        
        self.cur_targ_pos = self.ee_pose()[1]

def main():
    parser = argparse.ArgumentParser(description="Connect to Franka Emika Panda robot")
    parser.add_argument("--ip", type=str, required=True, help="IP address of the Franka Emika Panda robot")
    args = parser.parse_args()

    controller = FrankaController(args.ip)
    R, t = controller.ee_pose()

    pygame.init()
    pygame.joystick.init()
    controller.apply_q(np.array([-0.08129526674747467, -0.09338368475437164, 0.02063392661511898, -2.354853630065918, 0.002519397297874093, 2.2613837718963623, 0.723608493804932]), duration=3.0)
    target_pos = controller.cur_targ_pos.copy()
    last_tp = time.time()
    dur = 0.0
    while True:
        move_vel = np.array([0.0, 0.0, 0.0])

        if pygame.joystick.get_count() > 0:
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
            
            pygame.event.pump()  # Process event queue
            
            axis_0 = joystick.get_axis(0)
            axis_1 = joystick.get_axis(1)

            axis_2 = (joystick.get_axis(2) + 1.0) * 0.5
            axis_5 = (joystick.get_axis(5) + 1.0) * 0.5
            move_vel[1] = axis_0 * 0.05  # Scale to max 0.1 m/s
            move_vel[0] = axis_1 * 0.05  # Invert Y axis
            move_vel[2] = (axis_5 - axis_2) * 0.05  # Scale to max 0.1 m/s

              # if button 1 pressed, terminate
            if joystick.get_button(1):
                print("Exiting joystick control.")
                break
        curr_time = time.time()
        dur = curr_time - last_tp
        last_tp = curr_time
        target_pos += move_vel * dur
        controller.apply_targ_pos(target_pos)

if __name__ == "__main__":
    main()