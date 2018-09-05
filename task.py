import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 1
#         self.action_high = 700
        self.action_high = 900        
        self.action_size = 4
        self.position_target_margin = 0.1
        
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 150.]) 
        if target_pos is None:
            print("WARNING: ASSUMING TARGET POSITION OF: {}".format(self.target_pos))

        self.DEBUG = False
            
    def _get_euclidean_distance(self, point1, point2):
        import math
        x_delta = point1[0] - point2[0]
        y_delta = point1[1] - point2[1]
        z_delta = point1[2] - point2[2]    
        distance = math.sqrt((x_delta) ** 2 + (y_delta) ** 2 + (z_delta) ** 2)
        return distance
        
        
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        
        # old reward function based on all 3 coordinates:
        # reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        
        # new reward function based on elevation (z-cooridnate, 3rd coordinate) alone:
        # reward = 1 / (1 + abs(self.sim.pose[2] - self.target_pos[2]))
        # reward = 1 - abs(self.sim.pose[2] - self.target_pos[2])

        distance = self._get_euclidean_distance(self.sim.pose, self.target_pos)        

        velocity_vect_length = self._get_euclidean_distance(self.sim.v, [0, 0, 0])

        angular_velocity_vect_length = self._get_euclidean_distance(self.sim.angular_v, [0, 0, 0])
        
#         reward = 1 - np.log(distance + 1)
        
        reward = 1 + (-0.2 * distance) + (-0.01 * velocity_vect_length) + (-0.5 * angular_velocity_vect_length)
        
        # print("  distance is {:.2f}, reward is {:.2f}".format(distance, reward))
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            last_reward = self.get_reward() 
            reward += last_reward
            pose_all.append(self.sim.pose)

        if self.DEBUG:
            print("\nDEBUG:       "
                  "ROTORS:  {:.0f}  {:.0f}  {:.0f}  {:.0f}    "
                  "VEL:  {:.1f}  {:.1f}  {:.1f}    "              
                  "POS:  {:.1f}  {:.1f}  {:.1f}    "
                  "ANG-VEL:  {:.1f}  {:.1f}  {:.1f}    "              
                  "R:  {:.1f}                   ".format(rotor_speeds[0], 
                                                         rotor_speeds[1],
                                                         rotor_speeds[2],
                                                         rotor_speeds[3], 
                                                         self.sim.v[0],
                                                         self.sim.v[1],
                                                         self.sim.v[2],
                                                         self.sim.pose[0],
                                                         self.sim.pose[1],
                                                         self.sim.pose[2],
                                                         self.sim.angular_v[0],
                                                         self.sim.angular_v[1],
                                                         self.sim.angular_v[2],
                                                         last_reward))
            
            
        next_state = np.concatenate(pose_all)

        if self._get_euclidean_distance(self.sim.pose, self.target_pos) < self.position_target_margin:
            done = True
            print("YEY - reached aprox. target! pos {} vs target {}".format(self.sim.pose[:3], self.target_pos))
        
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state