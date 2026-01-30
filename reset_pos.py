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

        state = self.robot.read_once()
        self.start_q = np.array(state.q)
        self.last_torque = None
        
        self.torque_log = []
        self.torque_diff_log = []
        self.pos_err_log = []
        self.pos_err = []

        self.current_pos_err = np.zeros(7)

    def ee_pose(self):
        state = self.robot.read_once()
        transform = np.array(state.O_T_EE).reshape(4, 4).T
        R = transform[:3, :3].copy()
        t = transform[:3, 3].copy()
        # R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]) @ R
        return R, t

    def apply_torque(self, tau_d):
        torque_command = pylibfranka.Torques(tau_d.tolist())
        torque_command.motion_finished = False
        self.active_controller.writeOnce(torque_command)

        if self.last_torque is not None:
            diff = np.array(tau_d) - np.array(self.last_torque)
            # max component absolute value of diff
            self.torque_diff_log.append(np.max(np.abs(diff)))
            self.torque_log.append(np.linalg.norm(tau_d))
        self.last_torque = tau_d
    
    def reset_to_pose(self, q_targ: np.ndarray, duration: float = 2.0):
        state, _ = self.active_controller.readOnce()
        start_q = np.array(state.q)
        elapsed_time = 0.0
        waiting = False
        wait_time_finish = 0.0

        while True:
            state, dur = self.active_controller.readOnce()

            coriolis = np.array(self.model.coriolis(state))
            q = np.array(state.q)
            dq = np.array(state.dq)

            elapsed_time += dur.to_sec()
            alpha = min(elapsed_time / duration, 1.0)
            q_desired = (1 - alpha) * self.start_q + alpha * q_targ

            position_error = q - q_desired

            self.pos_err_log.append(np.linalg.norm(position_error))
            self.pos_err.append(position_error)

            tau_task = -self.joint_stiffness * position_error - self.joint_damping * dq

            # norm(pos_err) < 0.1
            # 

            tau_d = tau_task + coriolis
            self.apply_torque(tau_d)

            if elapsed_time >= duration and not waiting:
                waiting = True
                wait_time_finish = elapsed_time + 0.0
                self.start_q = q_targ.copy()
            if waiting and elapsed_time >= wait_time_finish:
                break

    def move_at_velocity(self, v_desired):
        state, dur = self.active_controller.readOnce()
        J = np.array(self.model.zero_jacobian(state)).reshape(6, 7)
        coriolis = np.array(self.model.coriolis(state))
        q = np.array(state.q)
        dq = np.array(state.dq)

        dx = J @ dq
        # print(f"Jacobian, {J.shape}:\n{J}\n, Linear Velocity: {dx[:3]} m/s\n Angular Velocity: {dx[3:]} rad/s")

        # pseudo-inverse of J
        J_pinv = np.linalg.pinv(J)
        # print(f"Pseudo-inverse of Jacobian, {J_pinv.shape}:\n{J_pinv}\n")
        dq_desired = J_pinv @ v_desired
        print(f"Desired joint velocities: {dq_desired}")

        dq_desired *= dur.to_sec()

        tau_task = self.joint_stiffness * dq_desired - self.joint_damping * dq

        tau_d = tau_task + coriolis
        torque_command = pylibfranka.Torques(tau_d.tolist())
        torque_command.motion_finished = False
        self.active_controller.writeOnce(torque_command)

    def show_torque_log(self):
        import matplotlib.pyplot as plt

        plt.figure()
        # plt.plot(self.torque_log, label="Torque Norm")
        plt.plot(self.torque_diff_log, label="Torque Difference Norm")
        # plt.plot(self.pos_err_log, label="Position Error Norm")

        # plot the norm of pos_err diffs of adjacent entries

        # pos_err_diffs = [np.linalg.norm(self.pos_err[i] - self.pos_err[i-1]) for i in range(1, len(self.pos_err))]
        # plt.plot(pos_err_diffs, label="Position Error Difference Norm")

        plt.xlabel("Time Step")
        plt.ylabel("Norm")
        plt.title("Torque Norms Over Time")
        plt.legend()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Connect to Franka Emika Panda robot")
    parser.add_argument("--ip", type=str, required=True, help="IP address of the Franka Emika Panda robot")
    args = parser.parse_args()

    controller = FrankaController(args.ip)
    R, t = controller.ee_pose()


    controller.reset_to_pose(np.array([-0.08129526674747467, -0.09338368475437164, 0.02063392661511898, -2.354853630065918, 0.002519397297874093, 2.2613837718963623, 0.723608493804932]), duration=3.0)
    # for i in range(10):
    #     targ_q = np.array([-0.0377, -0.1855,  0.0390, -2.3545,  0.0050,  2.1693,  0.7897])
    #     # add random angle of 10 degrees to every joint
    #     targ_q += np.deg2rad(np.random.uniform(-10.0, 10.0, size=(7,)))
    #     controller.reset_to_pose(targ_q, duration=3.0)
    # controller.reset_to_pose(np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0]), duration=3.0)
    # controller.reset_to_pose(np.array([-0.0377, -0.1855,  0.0390, -2.3545,  0.0050,  2.1693,  0.7897]), duration=3.0)
    # controller.reset_to_pose(np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0]), duration=3.0)
    # controller.reset_to_pose(np.array([-0.0377, -0.1855,  0.0390, -2.3545,  0.0050,  2.1693,  0.7897]), duration=3.0)
    # controller.reset_to_pose(np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0]), duration=3.0)
    # controller.reset_to_pose(np.array([-0.0377, -0.1855,  0.0390, -2.3545,  0.0050,  2.1693,  0.7897]), duration=3.0)
    # controller.reset_to_pose(np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0]), duration=3.0)
    # controller.reset_to_pose(np.array([-0.0377, -0.1855,  0.0390, -2.3545,  0.0050,  2.1693,  0.7897]), duration=3.0)

    # print(f"End-effector position: {rotation_matrix_to_axis_angle(R), t}")

    # controller.show_torque_log()

    # for _ in trange(100000):
    #     controller.move_at_velocity(np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0]))
    # state = controller.robot.read_once()
    # print(f"Joint positions: {state.q}, err: {(np.array(state.q) - np.array([-0.0377, -0.1855,  0.0390, -2.3545,  0.0050,  2.1693,  0.7897])) * 180.0 / np.pi}")


# End-effector position: ((array([-9.99928081e-01, -5.06692273e-04,  1.19822936e-02]), 3.1319807905709243), array([0.4782218 , 0.00471156, 0.28644663]))
# Joint positions: [-0.015996411442756653, -0.1449327915906906, 0.021749958395957947, -2.3437929153442383, 0.015601209364831448, 2.1745598316192627, 0.7816110849380493], err: [ 1.24352402  2.32432983 -0.98835458  0.61347076  0.60740455  0.30136615
#  -0.46346069]

if __name__ == "__main__":
    main()