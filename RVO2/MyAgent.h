/**
 * \file       MyAgent.h
 * \brief      Contains the Agent class.
 */
#ifndef RVO_MYAGENT_H_
#define RVO_MYAGENT_H_

#include "Definitions.h"
#include "MyRVOSim.h"

namespace RVO {
	/**
	 * \brief      Defines an agent in the simulation.
	 */
	class Agent {
	private:
		/**
		 * \brief      Constructs an agent instance.
		 * \param      sim             The simulator instance.
		 */
		explicit Agent(RVOSimulator *sim);

		/**
		 * \brief      Computes the neighbors of this agent.
		 */
		void computeNeighbors();
		void insertAgentNeighbor(const Agent *agent, float &rangeSq);
		/**
		 * \brief      Computes the new velocity of this agent.
		 */
		void computeNewVelocity();

		/**
		 * \brief      Updates the two-dimensional position and two-dimensional
		 *             velocity of this agent.
		 */
		void update();


        // \brief:                   The properties of a agent.
        // \param   position_:       Current position of directed agent.
        // \param   velocity_:       Current velocity of directed agent.    
		// \param   prefVelocity:    Prefer velocity of directed anget.
        // \param   radius_:         Rdius of directed agent.
        // \param   agentNeighbors_: Neighbors which likely to collide with directed agent.
        // \param   newVelocity_:	 New velocity of directed agent.
		// \param	maxSpeed_;		 Maximal speed of a agent.
		// \param	neighborDist_	 Maximal distance between two agents in the system.
        Vector2 position_;
        Vector2 velocity_;
        Vector2 prefVelocity_;
        float radius_;
        std::vector<std::pair<float, const Agent *> > agentNeighbors_;
		Vector2 newVelocity_;
		std::vector<Line> orcaLines_;
		RVOSimulator *sim_;
		float maxSpeed_;
		float timeHorizon_;
		float neighborDist_;
		size_t maxNeighbors_;


		friend class KdTree;
		friend class RVOSimulator;
	};

	/**
	 * \relates    Agent
	 * \brief      Solves a one-dimensional linear program on a specified line
	 *             subject to linear constraints defined by lines and a circular
	 *             constraint.
	 * \param      lines         Lines defining the linear constraints.
	 * \param      lineNo        The specified line constraint.
	 * \param      radius        The radius of the circular constraint.( for velocity)
	 * \param      optVelocity   The optimization velocity.
	 * \param      directionOpt  True if the direction should be optimized.
	 * \param      result        A reference to the result of the linear program.
	 * \return     True if successful.
	 */
	bool linearProgram1(const std::vector<Line> &lines, size_t lineNo,
						float radius, const Vector2 &optVelocity,
						bool directionOpt, Vector2 &result);

	/**
	 * \relates    Agent
	 * \brief      Solves a two-dimensional linear program subject to linear
	 *             constraints defined by lines and a circular constraint.
	 * \param      lines         Lines defining the linear constraints.
	 * \param      radius        The radius of the circular constraint.	(for velocity)
	 * \param      optVelocity   The optimization velocity.
	 * \param      directionOpt  True if the direction should be optimized.
	 * \param      result        A reference to the result of the linear program.
	 * \return     The number of the line it fails on, and the number of lines if successful.
	 */
	size_t linearProgram2(const std::vector<Line> &lines, float radius,
						  const Vector2 &optVelocity, bool directionOpt,
						  Vector2 &result);

	/**
	 * \relates    Agent
	 * \brief      Solves a two-dimensional linear program subject to linear
	 *             constraints defined by lines and a circular constraint.
	 * \param      lines         Lines defining the linear constraints.
	 * \param      numObstLines  Count of obstacle lines.	>>> set numObstline == 0 in simulator
	 * \param      beginLine     The line on which the 2-d linear program failed.
	 * \param      radius        The radius of the circular constraint.		(for velocitys)
	 * \param      result        A reference to the result of the linear program.
	 */
	void linearProgram3(const std::vector<Line> &lines, size_t numObstLines, size_t beginLine,
						float radius, Vector2 &result);
}

#endif /* RVO_AGENT_H_ */
