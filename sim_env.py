import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

class G1LabEnv(gym.Env):
    """
    Custom Gymnasium environment that loads a MuJoCo MJCF model directly.
    Demonstrates the Unitree G1 robot with a lab bench and target vial.
    """
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, xml_file, render_mode=None):
        self.xml_file = xml_file
        self.render_mode = render_mode
        
        self.model = mujoco.MjModel.from_xml_path(self.xml_file)
        self.data = mujoco.MjData(self.model)
        
        self.viewer = None
        
        # Action space = number of actuators in the G1
        nu = self.model.nu
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(nu,), dtype=np.float32)
        
        # Observation space = positions + velocities
        nq = self.model.nq
        nv = self.model.nv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(nq + nv,), dtype=np.float64)
        
        # Cache control ranges for scaling actions
        self.ctrl_range = self.model.actuator_ctrlrange
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs(), {}
        
    def step(self, action):
        # Scale incoming [-1, 1] action to the actual control ranges of the G1
        action = np.clip(action, -1.0, 1.0)
        low, high = self.ctrl_range[:, 0], self.ctrl_range[:, 1]
        action = low + (action + 1.0) * 0.5 * (high - low)
        
        self.data.ctrl[:] = action
        
        # Step the physics engine (e.g., 5 simulation steps per environment step)
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
            
        if self.render_mode == "human":
            self.render()
            
        gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
        vial_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_vial")
        
        gripper_pos = self.data.xpos[gripper_id]
        vial_pos = self.data.xpos[vial_id]
        gripper_mat = self.data.xmat[gripper_id].reshape(3, 3)
        
        # 1. Distance reward (exponentially increasing as distance decreases)
        dist = np.linalg.norm(gripper_pos - vial_pos)
        distance_reward = np.exp(-10.0 * dist)
        
        # 2. Orientation reward (gripper z-direction aligned with vial z-direction)
        gripper_z = gripper_mat[:, 2] # Local z-axis of gripper
        desired_z = np.array([0, 0, -1]) # Assuming we want to point straight down at the vial
        orientation_reward = np.dot(gripper_z, desired_z) * 0.5 + 0.5 # mapped to [0, 1]
        
        # 3. Action and velocity penalties for smooth motion
        action_penalty = -0.01 * np.sum(np.square(action))
        velocity_penalty = -0.001 * np.sum(np.square(self.data.qvel))
        
        # 4. Collision / Stability penalty
        vial_z = vial_pos[2]
        collision_penalty = 0.0
        terminated = False
        
        # Bench top is roughly 0.8 (0.4 pos z + 0.4 size). If vial drops below 0.8, it fell.
        if vial_z < 0.8:
            collision_penalty = -100.0
            terminated = True
            
        # 5. Grasp success
        grasp_success = 0.0
        if dist < 0.05 and vial_z > 0.86: # Lifted slightly off the bench
            grasp_success = 100.0
            terminated = True
            
        reward = distance_reward + orientation_reward + action_penalty + velocity_penalty + collision_penalty + grasp_success
        truncated = False
        
        obs = self._get_obs()
        return obs, reward, terminated, truncated, {
            "distance": dist,
            "dist_rew": distance_reward,
            "ori_rew": orientation_reward,
            "act_pen": action_penalty,
            "vel_pen": velocity_penalty,
            "col_pen": collision_penalty,
            "grasp": grasp_success
        }
        
    def _get_obs(self):
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])
        
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

if __name__ == "__main__":
    import os
    import time

    # Point to the custom G1 lab scene we just built
    xml_path = os.path.join("models", "unitree_mujoco-main", "unitree_robots", "g1", "g1_lab_scene.xml")
    
    print(f"Loading custom G1 Env from {xml_path}...")
    env = G1LabEnv(xml_path, render_mode="human")
    obs, info = env.reset()
    
    print("Environment loaded. Running simulation loop to verify physics and rendering...")
    # Render the initial state for a moment before starting
    env.render()
    time.sleep(1)
    
    for _ in range(500):
        # Sample random joint actions for demonstration
        action = env.action_space.sample() * 0.1 # Keep actions small so it doesn't instantly collapse violently
        obs, reward, term, trunc, info = env.step(action)
        
        if term or trunc:
            env.reset()
            
    env.close()
    print("Simulation verification completed.")
