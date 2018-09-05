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
        self.position_target_margin = 0.5
        self.init_pose = init_pose
                
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 150.]) 
        if target_pos is None:
            print("WARNING: ASSUMING TARGET POSITION OF: {}".format(self.target_pos))

        self.distance_to_travel = self._get_euclidean_distance(init_pose, self.target_pos)
        print("Distance to travel: {}".format(self.distance_to_travel))
            
        self.DEBUG = False
            
    def _get_euclidean_distance(self, point1, point2):
        import math
        x_delta = point1[0] - point2[0]
        y_delta = point1[1] - point2[1]
        z_delta = point1[2] - point2[2]    
        distance = float(math.sqrt((x_delta ** 2) + (y_delta ** 2) + (z_delta ** 2)))
        return distance
        
        
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        
        # reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()        
        # new reward function based on elevation (z-cooridnate, 3rd coordinate) alone:
        # reward = 1 / (1 + abs(self.sim.pose[2] - self.target_pos[2]))
        # reward = 1 - abs(self.sim.pose[2] - self.target_pos[2])
        # reward = 1 - np.log(distance + 1)

        distance = self._get_euclidean_distance(self.sim.pose[:3], self.target_pos)        
        velocity_vect_length = self._get_euclidean_distance(self.sim.v, [0, 0, 0])
        angular_velocity_vect_length = self._get_euclidean_distance(self.sim.angular_v, [0, 0, 0])
        
        distance_loss = -2.0 * distance / float(self.distance_to_travel)
        velocity_vect_loss = -0.0 * velocity_vect_length / 900.0
        angular_vel_loss = -1.0 * angular_velocity_vect_length / 30.0   
        time_gain = 0.5 * self.sim.time / 5.0
        
        # aiming for max reward when we get to: np.tanh(2):
        reward = np.tanh(2 + distance_loss + angular_vel_loss + time_gain)
        
#         reward = np.tanh(
#             2 + 
#             (-3.0 * (distance / 10.)) + 
#             (-0.1 * (velocity_vect_length / 500.)) + 
#             (-1.0 * (angular_velocity_vect_length / 30.))
#         )
        
        if self.DEBUG and int(self.sim.time * 10) % 10 == 0:
            print("@ {:.1f} | D: {:.1f} ({:.1f})  V: {:.1f} ({:.1f})  AV: {:.1f} ({:.1f})   TIMEGAIN: ({:.1f})   = R {:.2f}"
                  .format(self.sim.time, distance, distance_loss, 
                          velocity_vect_length, velocity_vect_loss, 
                          angular_velocity_vect_length, angular_vel_loss, 
                          time_gain, reward))
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        
        reward = 0
        pose_all = []
        reaced_target_pos = False
        for _ in range(self.action_repeat):

            # test code to make sure that uniform rotor speeds can cause the copter to lift up and for the velocities stay fairly 
            # uniform:
#             done = self.sim.next_timestep([600, 600, 600, 600])
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities

            distance = self._get_euclidean_distance(self.sim.pose[:3], self.target_pos)

            last_reward = self.get_reward() 
            reward += last_reward
            pose_all.append(self.sim.pose)
            
            if distance <= self.position_target_margin:
                reaced_target_pos = True

        if reaced_target_pos:
            print("\nYEY! Reached the ~target: pos {} vs target {}\n".format(self.sim.pose[:3], self.target_pos))
            done = True
            
        next_state = np.concatenate(pose_all)
        
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state