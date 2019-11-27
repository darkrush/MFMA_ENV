/*
 * KdTree.h
 * RVO2 Library
 */

#ifndef RVO_KD_TREE_H_
#define RVO_KD_TREE_H_

/**
 * \file       KdTree.h
 * \brief      Contains the KdTree class.
 */

#include "Definitions.h"

namespace RVO {
	/**
	 * \brief      Defines <i>k</i>d-trees for agents and static obstacles in the
	 *             simulation.
	 */
	class KdTree {
	private:
		/**
		 * \brief      Defines an agent <i>k</i>d-tree node.
		 */
		class AgentTreeNode {
		public:
			/**
			 * \brief      The beginning node number.
			 */
			size_t begin;

			/**
			 * \brief      The ending node number.
			 */
			size_t end;

			/**
			 * \brief      The left node number.
			 */
			size_t left;

			/**
			 * \brief      The maximum x-coordinate.
			 */
			float maxX;

			/**
			 * \brief      The maximum y-coordinate.
			 */
			float maxY;

			/**
			 * \brief      The minimum x-coordinate.
			 */
			float minX;

			/**
			 * \brief      The minimum y-coordinate.
			 */
			float minY;

			/**
			 * \brief      The right node number.
			 */
			size_t right;
		};

		/**
		 * \brief      Constructs a <i>k</i>d-tree instance.
		 * \param      sim             The simulator instance.
		 */
		explicit KdTree(RVOSimulator *sim);

		/**
		 * \brief      Destroys this kd-tree instance.
		 */
		~KdTree();

		/**
		 * \brief      Builds an agent <i>k</i>d-tree.
		 */
		void buildAgentTree();

		void buildAgentTreeRecursive(size_t begin, size_t end, size_t node);

		/**
		 * \brief      Computes the agent neighbors of the specified agent.
		 * \param      agent           A pointer to the agent for which agent
		 *                             neighbors are to be computed.
		 * \param      rangeSq         The squared range around the agent.
		 */
		void computeAgentNeighbors(Agent *agent, float &rangeSq) const;


		void queryAgentTreeRecursive(Agent *agent, float &rangeSq,
									 size_t node) const;

		/**
		 * \brief      Queries the visibility between two points within a
		 *             specified radius.
		 * \param      q1              The first point between which visibility is
		 *                             to be tested.
		 * \param      q2              The second point between which visibility is
		 *                             to be tested.
		 * \param      radius          The radius within which visibility is to be
		 *                             tested.
		 * \return     True if q1 and q2 are mutually visible within the radius;
		 *             false otherwise.
		 */
		bool queryVisibility(const Vector2 &q1, const Vector2 &q2,
							 float radius) const;

		bool queryVisibilityRecursive(const Vector2 &q1, const Vector2 &q2,
									  float radius) const;

		std::vector<Agent *> agents_;
		std::vector<AgentTreeNode> agentTree_;
		RVOSimulator *sim_;

		static const size_t MAX_LEAF_SIZE = 10;

		friend class Agent;
		friend class RVOSimulator;
	};
}

#endif /* RVO_KD_TREE_H_ */
