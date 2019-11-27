#ifndef RVO_MYRVOSIM_H_
#define RVO_MYRVOSIM_H_

/**
 * \file       RVOSimulator.h
 * \brief      Contains the RVOSimulator class.
 */

#include <cstddef>
#include <limits>
#include <vector>

#include "Vector2.h"
#include <iostream>

namespace RVO{
	/**
	 * \brief       Error value.
	 *
	 * A value equal to the largest unsigned integer that is returned in case
	 * of an error by functions in RVO::RVOSimulator.
	 */
	const size_t RVO_ERROR = std::numeric_limits<size_t>::max();

    // \brief:   Defines a line with direction.
    class Line {
    public:
        // \brief: A point on the directed line.
        Vector2 point;
        
        // \brief: Direction of the directed line.
        Vector2 direction;
    };

    class Agent;
    class KdTree;

    class RVOSimulator {
    public:
        // \brief:  Constructs a simulator instance.
        RVOSimulator();
        ~RVOSimulator();


        // \brief:              Add a agent with directed properties in the simulator.
        // \param   xposition   X-coordinate axis position of agent.
        // \param   yposition   Y-coordinate axis position of agent.
        // \param   xvelocity   X-coordinate axis Velocity of agent.
        // \param   yvelocity   Y-coordinate axis Velocity of agent.
        // \param   xpreferv    X-coordinate axis prefer Velocity of agent.
        // \param   ypreferv    Y-coordinate axis prefer Velocity of agent.
        // \param   radius      Radius of a agent.
       // size_t addAgent(float xposition, float yposition, float xvelocity, float  yvelocity,float xpreferv, float ypreferv, 
        //                float radius, float maxSpeed, float timeHorizon, float neighborDist);
		//size_t addAgent(Vector2, Vector2, Vector2, float, float, float, float);
        size_t addAgent(float, float, float, float, float, float, float, float, float, float);
        
        /**
		 //\brief      Returns the count of agents in the simulation.
		 //\return     The count of agents in the simulation.
		 */
		size_t getNumAgents() const;

		/**
		 * \brief      Returns the two-dimensional position of a specified
		 *             agent.
		 * \param      agentNo         The number of the agent whose
		 *                             two-dimensional position is to be retrieved.
		 * \return     The present two-dimensional position of the (center of the)
		 *             agent.
		 */
		const Vector2 &getAgentPosition(size_t agentNo) const;

//        const Vector2 &getAgentVelocity(size_t agentNo) const;

        float* getAgentVelocity(size_t agentNo);
		/**
		 * \brief      Returns the radius of a specified agent.
		 * \param      agentNo         The number of the agent whose radius is to
		 *                             be retrieved.
		 * \return     The present radius of the agent.
		 */
	//	float getAgentRadius(size_t agentNo);

        void setStep(float timeStep);

        void doStep();
        
        // \brief:          Properties of a directed RVOSimulator.
        private:
            std::vector<Agent *> agents_;
            KdTree *kdTree_;
            float timeStep_;

            friend class Agent;
            friend class KdTree;
    };

}

#endif