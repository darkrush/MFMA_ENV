//  \file       MyRVO.cpp
//  \brief      Definition of function which used for compute new velocity of agents.
#include "MyRVOSim.h"
#include "Vector2.h"
#include <vector>

extern "C"{ // Use extern "C" to make sure compile C++ functions the way C does
void AgentNewVel(const int Agentnum, const float pos_x[], const float pos_y[], float velx[], float vely[], const float preVx[], const float preVy[],
                 const float radius[], const float maxSpeed[], const float timeHorizon, const float neighborDist, const float timeStep) 
{
    RVO::RVOSimulator *sim = new RVO::RVOSimulator();
    float *newvelocity;
    // \brief   Set time step of simulator.
    sim->setStep(timeStep);
    // \brief   Add robot state in the simulator with the input data.
    for (size_t i = 0; i < Agentnum; ++i) {
        sim->addAgent(pos_x[i], pos_y[i], velx[i], vely[i], preVx[i], preVy[i], radius[i], maxSpeed[i], timeHorizon, neighborDist);
    }
    //  \brief  Perform this step to calculate the robot's new speed.
    sim->doStep();
    //  \brief  Get the updated robot speed and output.
    for (size_t i = 0; i < Agentnum; i++) {
        newvelocity = sim->getAgentVelocity(i);
        velx[i] = newvelocity[0];
        vely[i] = newvelocity[1];   }
    delete sim;
}
} // extern "C"