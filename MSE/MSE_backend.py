import sys 
sys.path.append("../../")
from MF_env import basic

from . import core
import time
import math
import numpy as np

import os


def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    #r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return (r, g, b)


class MSE_backend(object):
    def __init__(self,scenario, dt = 0.1,window_scale = 1.0, multi_thread = False):
        self.window_scale = window_scale
        self.agent_groups = scenario['agent_groups']
        self.use_gui = scenario['common']['use_gui']
        self.cam_range = 4
        self.viewer = None
        prop_list = []
        for (_,agent_group) in self.agent_groups.items():
            for agent_prop in agent_group:
                temp_prop = core.AgentProp()
                temp_prop.R_safe = agent_prop['R_safe']
                temp_prop.R_reach = agent_prop['R_reach']
                temp_prop.L_car = agent_prop['L_car']
                temp_prop.W_car = agent_prop['W_car']
                temp_prop.L_axis = agent_prop['L_axis']
                temp_prop.R_laser = agent_prop['R_laser']
                temp_prop.N_laser = agent_prop['N_laser']
                temp_prop.K_vel = agent_prop['K_vel']
                temp_prop.K_phi = agent_prop['K_phi']
                temp_prop.init_x = agent_prop['init_x']
                temp_prop.init_y = agent_prop['init_y']
                temp_prop.init_theta = agent_prop['init_theta']
                temp_prop.init_vel_b = agent_prop['init_vel_b']
                temp_prop.init_phi = agent_prop['init_phi']
                temp_prop.init_movable = agent_prop['init_movable']
                temp_prop.init_target_x = agent_prop['init_target_x']
                temp_prop.init_target_y = agent_prop['init_target_y']
                prop_list.append(temp_prop)
        self.agent_number = len(prop_list)
        self.world = core.World(prop_list,dt)
        self._reset_render()
    
    def _reset_render(self):
        self.agent_geom_list = None
    
    def render(self,time = '0', mode='human'):
        if self.viewer is None:
            from . import rendering 
            self.viewer = rendering.Viewer(800,800)
        self.agents = []
        self.color_list = []
        for idx in range(self.agent_number):
            self.world.update_laser_state(idx)
            self.color_list.append(hsv2rgb(360.0/self.agent_number*idx,1.0,1.0))
            self.agents.append(self.world.get_agent(idx))
        # create rendering geometry
        self.agent_geom_list = None
        if self.agent_geom_list is None:
            # import rendering only if we need it (and don't import for headless machines)
            from . import rendering
            self.viewer.set_bounds(0-self.cam_range, 0+self.cam_range, 0-self.cam_range, 0+self.cam_range)
            self.agent_geom_list = []
            for idx,agent in enumerate(self.agents):
                enable =  agent.state.enable
                agent_geom = {}
                total_xform = rendering.Transform()
                agent_geom['total_xform'] = total_xform
                agent_geom['laser_line'] = []

                geom = rendering.make_circle(agent.prop.R_reach)
                geom.set_color(*self.color_list[idx],alpha  = 1.0 if enable else 0.0)
                xform = rendering.Transform()
                geom.add_attr(xform)
                agent_geom['target_circle']=(geom,xform)

                N = agent.prop.N_laser
                for idx_laser in range(N):
                    theta_i = idx_laser*math.pi*2/N
                    #d = agent.R_laser
                    d = 1
                    end = (math.cos(theta_i)*d, math.sin(theta_i)*d)
                    geom = rendering.make_line((0, 0),end)
                    geom.set_color(0.0,1.0,0.0,alpha = 0.5 if enable else 0.0)
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.add_attr(total_xform)
                    agent_geom['laser_line'].append((geom,xform))
                
                half_l = agent.prop.L_car/2.0
                half_w = agent.prop.W_car/2.0
                geom = rendering.make_polygon([[half_l,half_w],[-half_l,half_w],[-half_l,-half_w],[half_l,-half_w]])
                geom.set_color(*self.color_list[idx],alpha = 0.4 if enable else 0.0)
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.add_attr(total_xform)
                agent_geom['car']=(geom,xform)

                geom = rendering.make_line((0,0),(half_l,0))
                geom.set_color(1.0,0.0,0.0,alpha = 1 if enable else 0.0)
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.add_attr(total_xform)
                agent_geom['front_line']=(geom,xform)
                
                geom = rendering.make_line((0,0),(-half_l,0))
                geom.set_color(0.0,0.0,0.0,alpha = 1 if enable else 0.0)
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.add_attr(total_xform)
                agent_geom['back_line']=(geom,xform)

                self.agent_geom_list.append(agent_geom)

            self.viewer.geoms = []
            for agent_geom in self.agent_geom_list:
                self.viewer.add_geom(agent_geom['target_circle'][0])
                for geom in agent_geom['laser_line']:
                    self.viewer.add_geom(geom[0])
                self.viewer.add_geom(agent_geom['car'][0])
                self.viewer.add_geom(agent_geom['front_line'][0])
                self.viewer.add_geom(agent_geom['back_line'][0])
        
        for agent,agent_geom in zip(self.agents,self.agent_geom_list):
            for idx,laser_line in enumerate(agent_geom['laser_line']):
                laser_line[1].set_scale(agent.laser_state[idx],agent.laser_state[idx]) 
            agent_geom['front_line'][1].set_rotation(agent.state.phi)
            agent_geom['target_circle'][1].set_translation(agent.state.target_x*self.window_scale,agent.state.target_y*self.window_scale)
            agent_geom['target_circle'][1].set_scale(self.window_scale,self.window_scale)
            agent_geom['total_xform'].set_scale(self.window_scale,self.window_scale)
            agent_geom['total_xform'].set_rotation(agent.state.theta)
            agent_geom['total_xform'].set_translation(agent.state.x*self.window_scale,agent.state.y*self.window_scale)
            
        return self.viewer.render(time,return_rgb_array = mode=='rgb_array')

    def step(self,step_number = 1):
        render_frams = step_number if step_number <= 40 else int(step_number / 10)
        total_frams = step_number
        while total_frams>=0:
            self.world.step(render_frams if render_frams < total_frams else total_frams)
            if self.use_gui :
                self.render(time = '%.2f'%self.world.get_total_time())
            total_frams-=render_frams

    def get_state(self):
        cobj_state = self.world.get_states()
        gstate =[]
        for gstate_idx in range(0,self.agent_number):
            state_c = cobj_state[gstate_idx]
            state_py = basic.AgentState()
            state_py.x = state_c.x
            state_py.y = state_c.y
            state_py.vel_b = state_c.vel_b
            state_py.theta = state_c.theta
            state_py.phi = state_c.phi
            state_py.movable = state_c.movable
            state_py.crash = state_c.crash
            state_py.reach = state_c.reach
            state_py.enable = state_c.enable
            state_py.target_x = state_c.target_x
            state_py.target_y = state_c.target_y
            gstate.append(state_py)
        return self.world.get_total_time(),gstate
    
    def set_state(self,state_list,enable_list = None,reset = False,total_time = None):
        if enable_list is None:
            enable_list = [True]* len(state_list)
        for idx,state in enumerate(state_list):
            if enable_list[idx]:
                cobj_state = core.AgentState()
                cobj_state.x = state.x
                cobj_state.y = state.y
                cobj_state.theta = state.theta
                cobj_state.target_x = state.target_x
                cobj_state.target_y = state.target_y
                cobj_state.vel_b = state.vel_b
                cobj_state.phi = state.phi
                cobj_state.movable = state.movable
                cobj_state.enable = state.enable
                cobj_state.reach = state.reach
                cobj_state.crash = state.crash
                self.world.set_state(idx,cobj_state)
        if reset:
            self.world.set_total_time(0)
            self._reset_render()
        if total_time is not None:
            self.world.set_total_time(total_time)

    def get_obs(self):
        obs = []
        for obs_idx in range(0,self.agent_number):
            obs_c = self.world.get_obs(obs_idx)
            obs_py = basic.Observation()
            theta = obs_c.pos_theta
            xt = obs_c.pos_target_x - obs_c.pos_x
            yt = obs_c.pos_target_y - obs_c.pos_y
            xt,yt = (xt*np.cos(theta)+yt*np.sin(theta),yt*np.cos(theta)-xt*np.sin(theta))
            #obs_py.pos = [xt,yt]
            #obs_py.pos = [obs_c.pos_x,obs_c.pos_y,obs_c.pos_theta,obs_c.pos_target_x,obs_c.pos_target_y]
            obs_py.pos = [(xt**2+yt**2)**0.5,math.atan2(yt,xt)]
            for i in range(len(obs_c.laser_data)):
                obs_py.laser_data.append(obs_c.laser_data[i])
            obs.append(obs_py)
        return obs

    def set_action(self,actions,enable_list= None):
        if enable_list is None:
            enable_list = [True]* len(actions)
        for idx,enable in enumerate(enable_list):
            if enable:
                self.world.set_action(idx,actions[idx].ctrl_vel,actions[idx].ctrl_phi)


    def close(self):
        pass