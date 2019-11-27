#include"core.h"

void Agent::set_state(AgentState s)
{
    state = s;
    //for(int i=0;i<prop.N_laser;i++) laser_state[i] = prop.R_laser;
};

bool Agent::check_AA_collisions(Agent agent_b)
{
    float min_dist = pow((prop.R_safe + agent_b.prop.R_safe),2);
    float ab_dist = pow((state.x - agent_b.state.x),2) + pow((state.y - agent_b.state.y),2);
    return ab_dist <= min_dist;
};

bool Agent::check_reach()
{
    float max_dist = pow(prop.R_reach,2);
    float at_dist = pow((state.x - state.target_x),2) + pow((state.y - state.target_y),2);
    return at_dist<=max_dist;
};

vector<float> Agent::laser_agent_agent(Agent agent_b)
{
    
    float R = prop.R_laser;
    int N = prop.N_laser;
    vector<float>l_laser(N,R);
    float o_pos[2] = {state.x,state.y};
    float oi_pos[2] = {agent_b.state.x,agent_b.state.y};
    float l1 = sqrt((o_pos[0]-oi_pos[0])*(o_pos[0]-oi_pos[0])+(o_pos[1]-oi_pos[1])*(o_pos[1]-oi_pos[1]));
    float l2 = R + sqrt(agent_b.prop.L_car*agent_b.prop.L_car + agent_b.prop.W_car*agent_b.prop.W_car)/2.0;
    if(l1>l2) return l_laser;
    float theta = this->state.theta;
    float theta_b = agent_b.state.theta;
    float cthb = cos(theta_b);
    float sthb = sin(theta_b);
    float half_l_shift[2] = {cthb*agent_b.prop.L_car/2.0,sthb*agent_b.prop.L_car/2.0}; 
    float half_w_shift[2] = {-sthb*agent_b.prop.W_car/2.0,cthb*agent_b.prop.W_car/2.0};
    float car_point[4][2];
    for(int idx = 0; idx<2;idx++){
        car_point[0][idx] = oi_pos[idx]+half_l_shift[idx]+half_w_shift[idx]-o_pos[idx];
        car_point[1][idx] = oi_pos[idx]-half_l_shift[idx]+half_w_shift[idx]-o_pos[idx];
        car_point[2][idx] = oi_pos[idx]-half_l_shift[idx]-half_w_shift[idx]-o_pos[idx];
        car_point[3][idx] = oi_pos[idx]+half_l_shift[idx]-half_w_shift[idx]-o_pos[idx];
    }
    for(int i=0;i<4;i++)
    {
        float start_point[2] = {car_point[i][0],car_point[i][1]};
        float end_point[2] = {car_point[(i+1)%4][0],car_point[(i+1)%4][1]};
        float tao_es[2] = {start_point[1]-end_point[1],end_point[0]-start_point[0]};
        float norm_temp = sqrt(tao_es[0]*tao_es[0]+tao_es[1]*tao_es[1]);
        tao_es[0] = tao_es[0]/norm_temp;
        tao_es[1] = tao_es[1]/norm_temp;
        if(fabs(start_point[0]*tao_es[0]+start_point[1]*tao_es[1])>R) continue;
        if((start_point[0]*end_point[1]-start_point[1]*end_point[0])<0)
        {
            float temp;
            temp = start_point[0];
            start_point[0] = end_point[0];
            end_point[0] = temp;
            temp = start_point[1];
            start_point[1] = end_point[1];
            end_point[1] = temp;
        }
        float theta_start = acos(start_point[0]/sqrt(start_point[0]*start_point[0]+start_point[1]*start_point[1]));
        if(start_point[1]<0) theta_start = 2*pi-theta_start;
        theta_start -= theta;
        float theta_end = acos(end_point[0]/sqrt(end_point[0]*end_point[0]+end_point[1]*end_point[1]));
        if(end_point[1]<0) theta_end = 2*pi-theta_end;
        theta_end -= theta;
        float laser_idx_start = theta_start/(2*pi/N);
        float laser_idx_end = theta_end/(2*pi/N);
        
        if(laser_idx_start > laser_idx_end) laser_idx_end += N;
        if(floor(laser_idx_end)-floor(laser_idx_start)==0) continue;
        laser_idx_start = ceil(laser_idx_start);
        laser_idx_end = floor(laser_idx_end);
        for(int laser_idx=laser_idx_start;laser_idx<laser_idx_end+1;laser_idx++)
        {
            int laser_idx_ = (laser_idx+N)%N;
            float x1 = start_point[0];
            float y1 = start_point[1];
            float x2 = end_point[0];
            float y2 = end_point[1];
            float theta_i = theta+laser_idx_*2*pi/N;
            float cthi = cos(theta_i);
            float sthi = sin(theta_i);
            float temp = (y1-y2)*cthi-(x1-x2)*sthi;
            float dist;
            if(fabs(temp)<=1e-10) dist = R;
            else dist = (x2*y1-x1*y2)/temp;
            if(dist>0) l_laser[laser_idx_] = min(l_laser[laser_idx_],dist);
        }
    }
    return l_laser;
};


World::World(vector<AgentProp> agent_list,float cfg = 0.1)
{
    agent_num = agent_list.size();
    for(int idx = 0; idx < agent_num; idx++){
        Agent temp_agent(agent_list[idx]);
        agents.push_back(temp_agent);
    }
    
    dt = cfg;
    total_time = 0;
};
vector<AgentState> World::get_states()
{
    vector<AgentState> state_list;
    for(int idx = 0 ;idx < agent_num; idx++)
    {
        state_list.push_back(agents[idx].state);
    }
    return state_list;
}
Observation World::get_obs(int obs_idx)
{
    update_laser_state(obs_idx);
    AgentState state = agents[obs_idx].state;
    Observation obs;
    obs.pos_x = state.x;
    obs.pos_y = state.y;
    obs.pos_theta = state.theta;
    obs.pos_target_x = state.target_x;
    obs.pos_target_y = state.target_y;
    obs.laser_data = agents[obs_idx].laser_state;
    return obs;
};

void World::step(int step_num)
{
    for (int step_idx = 0; step_idx< step_num;step_idx++)
    {
        apply_action();
        integrate_state();
        check_collisions();
        check_reach();
        total_time += dt;
    }
};

float clip(float x , float l,float r){
    return (x>l)?((x<r)?(x):(r)):(l);
}

void World::apply_action()
{
    float movable_coef = 0;
    for(int i=0;i<agent_num;i++)
    {
        if(agents[i].state.movable) movable_coef =1;
        else movable_coef =0;
        agents[i].state.vel_b = clip(agents[i].action.ctrl_vel,-1,1)*agents[i].prop.K_vel*movable_coef;
        agents[i].state.phi   = clip(agents[i].action.ctrl_phi,-1,1)*agents[i].prop.K_phi*movable_coef;
    }
};

void World::update_laser_state(int idx_a)
{
    if(agents[idx_a].state.enable==false) return;
    for(int i=0;i<agents[idx_a].prop.N_laser;i++) agents[idx_a].laser_state[i] = agents[idx_a].prop.R_laser; 
    for(int idx_b=0;idx_b<agent_num;idx_b++)
    {
        if(idx_a==idx_b) continue;
        if(agents[idx_b].state.enable==false) continue;
        vector<float>l_laser = agents[idx_a].laser_agent_agent(agents[idx_b]);
        for(int j=0;j<agents[idx_a].prop.N_laser;j++) agents[idx_a].laser_state[j] = min(agents[idx_a].laser_state[j],l_laser[j]);
    }
};

void World::integrate_state()
{
    for(int i=0;i<agent_num;i++)
    {
        if(agents[i].state.enable==false) continue;
        if(agents[i].state.movable==false) continue;
        float _phi = agents[i].state.phi;
        float _vb = agents[i].state.vel_b;
        float _theta = agents[i].state.theta;
        float sth = sin(_theta);
        float cth = cos(_theta);
        float _L = agents[i].prop.L_axis;
        float _xb = agents[i].state.x-cth*_L/2.0;
        float _yb = agents[i].state.y-sth*_L/2.0;
        float tphi = tan(_phi);
        float _omega = _vb/_L*tphi;
        float _delta_theta = _omega*dt;
        float _rb,_delta_tao,_delta_yeta;
        if(fabs(_phi)>0.00001)
        {
            _rb = _L/tphi;
            _delta_tao = _rb*(1-cos(_delta_theta));
            _delta_yeta = _rb*sin(_delta_theta);
        }
        else
        {
            _delta_tao = _vb*dt*(_delta_theta/2.0);
            _delta_yeta = _vb*dt*(1-_delta_theta*_delta_theta/6.0);
        }
        _xb += _delta_yeta*cth - _delta_tao*sth;
        _yb += _delta_yeta*sth + _delta_tao*cth;
        _theta += _delta_theta;
        float fdec = (_theta/pi) - (int)(_theta/pi);
        _theta = (int)(_theta/pi)%2*pi+fdec*pi;
        agents[i].state.x = _xb+cos(_theta)*_L/2.0;
        agents[i].state.y = _yb+sin(_theta)*_L/2.0;
        agents[i].state.theta = _theta;
    }
};

void World::check_collisions()
{
    for(int ia=0;ia<agent_num;ia++)
    {
        if(agents[ia].state.enable==false) continue;
        if(agents[ia].state.crash) continue;
        for(int ib=0;ib<agent_num;ib++)
        {
            if(ia==ib) continue;
            if(agents[ib].state.enable==false) continue;
            if(agents[ia].check_AA_collisions(agents[ib]))
            {
                agents[ia].state.crash = true;
                agents[ia].state.movable = false;
                break;
            }
        }
    }
};

void World::check_reach()
{
    for(int i=0;i<agent_num;i++)
    {
        if(agents[i].state.enable==false) continue;
        bool reach = agents[i].check_reach();
        if(reach==true)
        {
            agents[i].state.reach = true;
            agents[i].state.movable = false;
        }
    }
};