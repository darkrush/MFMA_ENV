#ifndef CORE_H
#define CORE_H
#include<iostream>
#include<cmath>
#include<vector>
#define pi 3.1415926
using namespace std;


struct AgentProp{
    float R_safe;
    float R_reach;
    float L_car;
    float W_car;
    float L_axis;
    float R_laser;
    int N_laser;
    float K_vel;
    double K_phi;
    float init_x;
    float init_y;
    float init_theta;
    float init_vel_b;
    float init_phi;
    bool init_movable;
    float init_target_x;
    float init_target_y;
};

struct AgentState{
    float x;
    float y;
    float vel_b;
    float theta;
    float phi;
    bool enable;
    bool movable;
    bool crash;
    bool reach;
    float target_x;
    float target_y;
};

struct Action{
    float ctrl_vel;
    float ctrl_phi;
    Action(){ctrl_phi = 0;ctrl_phi = 0;};
};

struct Observation{
    float pos_x;
    float pos_y;
    float pos_theta;
    float pos_target_x;
    float pos_target_y;
    vector<float>laser_data;   //float*n
};



class Agent
{
    public:
        Agent(AgentProp prop){this->prop = prop;AgentState state = AgentState();Action action = Action();this->laser_state=vector<float> (prop.N_laser, prop.R_laser);};
        void set_state(AgentState);
        bool check_AA_collisions(Agent);        
        bool check_reach();
        vector<float> laser_agent_agent(Agent);
        AgentProp prop;
        AgentState state;
        Action action;
        vector<float>laser_state;
};



class World
{
    public:
        World(vector<AgentProp>, float);
        void reset_total_time(){total_time = 0;};

        void set_action(int action_idx,float vel,float phi){agents[action_idx].action.ctrl_vel = vel;agents[action_idx].action.ctrl_phi = phi;};
        void set_state(int state_idx,AgentState state){agents[state_idx].state=state;};
        
        AgentState get_state(int gstate_idx){return agents[gstate_idx].state;};
        vector<AgentState> get_states();
        Observation get_obs(int);
        Agent get_agent(int agent_idx){return agents[agent_idx];};
        float get_total_time(){return total_time;};
        float set_total_time(float time){total_time = time;};
        
        void step(int);
        void update_laser_state(int);
        
        
    //private:
        void apply_action();
        void integrate_state();
        void check_collisions();
        void check_reach();
        vector<Agent> agents;
        int agent_num;
        float total_time;
        float dt;
};
#endif 
