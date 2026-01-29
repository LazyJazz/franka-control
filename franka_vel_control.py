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
        joint_damping = [2.0 * np.sqrt(k) for k in joint_stiffness]

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

        self.current_pos_err = np.zeros(7)
        R, t = self.ee_pose()
        self.cur_targ_pos = t
        self.dur = 0.0

        self.torque_diff_log = []
        self.last_torque = np.zeros(7)

    def ee_pose(self):
        state = self.robot.read_once()
        transform = np.array(state.O_T_EE).reshape(4, 4).T
        R = transform[:3, :3].copy()
        t = transform[:3, 3].copy()
        R = R @ np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        t += np.array([-0.00112884, -0.00100483, 0.10032279])

        return R, t

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

    def apply_dq(self, dq_d: np.ndarray):
        state, dur = self.active_controller.readOnce()
        self.dur += dur.to_sec()
        # make dq normalized to have max norm of 0.1
        q = np.array(state.q)
        dq = np.array(state.dq)
        dq_d = dq_d - dq

        max_norm = 0.1
        norm_dq_d = np.linalg.norm(dq_d)
        if norm_dq_d > max_norm:
            dq_d = dq_d / norm_dq_d * max_norm
        dq_err = dq_d - self.current_pos_err

        # make dq_err have max norm of 0.001
        max_err_norm = 0.001
        norm_dq_err = np.linalg.norm(dq_err)
        if norm_dq_err > max_err_norm:
            dq_err = dq_err / norm_dq_err * max_err_norm

        self.current_pos_err += dq_err

        coriolis = np.array(self.model.coriolis(state))

        tau_task = self.joint_stiffness * self.current_pos_err - self.joint_damping * dq
        tau_d = tau_task + coriolis

        self.apply_torque(tau_d)

    def apply_vel(self, vel: np.ndarray):
        state, dur = self.active_controller.readOnce()
        self.dur += dur.to_sec()
        J = np.array(self.model.zero_jacobian(state)).reshape(7, 6).T
        # pseudo-inverse of J
        J_pinv = np.linalg.pinv(J)
        dq_desired = J_pinv @ vel
        self.apply_dq(dq_desired)

    def apply_linear_vel(self, vel: np.ndarray):
        # make vel have max norm of 0.1
        max_norm = 0.1
        norm_vel = np.linalg.norm(vel)
        if norm_vel > max_norm:
            vel = vel / norm_vel * max_norm
        R, t = self.ee_pose()

        # self.cur_targ_pos = t + vel
        self.cur_targ_pos += vel * self.dur
        self.dur = 0.0
        axis, angle = rotation_matrix_to_axis_angle(R)
        vel_full = np.zeros(6)
        vel_full[:3] = (self.cur_targ_pos - t) * 5.0
        vel_full[3:] = -axis * angle * 5000.0
        # make vel_full[3:] have max norm of 0.1
        max_rot_norm = 0.1
        norm_rot = np.linalg.norm(vel_full[3:])
        if norm_rot > max_rot_norm:
            vel_full[3:] = vel_full[3:] / norm_rot * max_rot_norm
        self.apply_vel(vel_full)


        

def main():
    parser = argparse.ArgumentParser(description="Connect to Franka Emika Panda robot")
    parser.add_argument("--ip", type=str, required=True, help="IP address of the Franka Emika Panda robot")
    args = parser.parse_args()

    controller = FrankaController(args.ip)
    R, t = controller.ee_pose()

    pygame.init()
    pygame.joystick.init()
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
            move_vel[1] = axis_0 * 0.03  # Scale to max 0.1 m/s
            move_vel[0] = axis_1 * 0.03  # Invert Y axis
            move_vel[2] = (axis_5 - axis_2) * 0.03  # Scale to max 0.1 m/s

              # if button 1 pressed, terminate
            if joystick.get_button(1):
                print("Exiting joystick control.")
                break

        controller.apply_linear_vel(move_vel)

        # R, t = controller.ee_pose()
        # print(f"End-effector position: {rotation_matrix_to_axis_angle(R), t}")

        # state = controller.robot.read_once()
        # print(f"Joint positions: {state.q}")

        # 0.44929960  0.012157003  0.14313839
        # 0.4492868, 0.01214779, 0.14305287

if __name__ == "__main__":
    main()