#include "MyRVOSim.h"

#include "MyAgent.h"
#include "KdTree.h"

namespace RVO {
    RVOSimulator::RVOSimulator(): kdTree_(NULL), timeStep_(0.25f)
    {
        kdTree_ = new KdTree(this);
    }

    RVOSimulator::~RVOSimulator()
    {
        for (size_t i = 0; i < agents_.size(); ++i)
        {
            delete agents_[i];
        }
        delete kdTree_;
    }
   // size_t RVOSimulator::addAgent(const std::vector<float> agentstate(8)){}
 /*   size_t RVOSimulator::addAgent(Vector2 position, Vector2 velocity, Vector2 prefVelocity,
                                  float radius, float maxSpeed, float timeHorizon, float neighborDist)
    {
        Agent *agent = new Agent(this);
        agent->position_ = position;
        agent->velocity_ = velocity;
        agent->prefVelocity_ = prefVelocity;
        agent->radius_ = radius;
        agent->maxSpeed_ = maxSpeed;
        agent->timeHorizon_ = timeHorizon;
        agent->neighborDist_ = neighborDist;
        agent->maxNeighbors_ = 10;


        agents_.push_back(agent);

        return agents_.size() - 1;
    }*/

    size_t RVOSimulator::addAgent(float xposition, float yposition, float xvelocity, float yvelocity, float xpreferv,
                                  float ypreferv, float radius, float maxSpeed, float timeHorizon, float neighborDist) 
    {
        Agent *agent = new Agent(this);
        agent->position_ = Vector2(xposition, yposition);
        agent->velocity_ = Vector2(xvelocity, yvelocity);
        agent->prefVelocity_ = Vector2(xpreferv, ypreferv);
        agent->radius_ = radius;
        agent->maxSpeed_ = maxSpeed;
        agent->timeHorizon_ = timeHorizon;
        agent->neighborDist_ = neighborDist;
        agent->maxNeighbors_ = 10;

        agents_.push_back(agent);
        return agents_.size() - 1;
    }


    size_t RVOSimulator::getNumAgents() const
	{
		return agents_.size();
	}

	const Vector2 &RVOSimulator::getAgentPosition(size_t agentNo) const
	{
		return agents_[agentNo]->position_;
	}
/**
    const Vector2 &RVOSimulator::getAgentVelocity(size_t agentNo) const
    {
    //  std::cout << "velocity of "<< agentNo << ":: "<<agents_[agentNo]->velocity_<<std::endl;
        return agents_[agentNo]->velocity_;
    }
**/
    float* RVOSimulator::getAgentVelocity(size_t agentNo)
    {
        float *velocity_return;
        velocity_return = new float[2];
        velocity_return[0] = agents_[agentNo]->velocity_.x();
        velocity_return[1] = agents_[agentNo]->velocity_.y();
        return velocity_return;
    }

/*	float RVOSimulator::getAgentRadius(size_t agentNo) const
	{
		return agents_[agentNo]->radius_;
	}*/

    void RVOSimulator::setStep(float timeStep) {
        timeStep_ = timeStep;
    }
    void RVOSimulator::doStep()
    {
        kdTree_->buildAgentTree();

        for (int i = 0; i < static_cast<int>(agents_.size()); ++i)
        {
            agents_[i]->computeNeighbors();
            agents_[i]->computeNewVelocity();
        }

        for (int i = 0; i < static_cast<int>(agents_.size()); ++i)
        {
            agents_[i]->update();
        }
    }
}