
// \file    MyRVO.h
// \brief   Includes header files and a function to calculate the robot's new speed.
#ifndef RVO_MYRVO_H
#define RVO_MYRVO_H

#include "MyRVOSim.h"
#include "Vector2.h"
#include <vector>

// \brief   Input current state of agents and compute their velocity for next step.
// \params   Position, velocity, preferred velocity, radius, maxspeed, timehorizon, neighborDist.
// \param Position: current position of agents.
// \param Velocity: current velocity of agents.
// \param Preferred veloctiy: prefer velocity move to target of each agent.
// \param Radius:   radius of agents,  different agents can have the same radius.
// \param MaxSpeed: maximum speed of agents, it's a scalar quantity, different agents can have same radius.
// \param timeHorizon:  The minimum time that different agents will not collide.
// \param neighborDist: the maximum distance between two different agents to judge if they're neighbors.
extern "C" {
 void AgentNewVel(const int, const float *, const float *, float* , float* , const float* , const float *,
                 const float *, const float *, const float , const float , const float );
}

#endif