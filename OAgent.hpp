/*
 * OAgent.hpp
 *
 *  Created on: May 26, 2015
 *      Author: Sina M. Baharlou (Sina.baharlou@gmail.com)
 */

#ifndef OAGENT_HPP_
#define OAGENT_HPP_


// -- Includes  --

#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <random>
#include <limits>

// -- OpenCV --
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


#include "Actions.hpp"
#include "RewardGenerator.hpp"
#include "QAgent.hpp"

// -- Definitions --

#define DEFAULT_POS 2,2
#define DEFAULT_LR	0.2
#define DEFAULT_GR	0.0
#define DEFAULT_DF	0.90
#define	DEFAULT_RF  0.0
#define DEFAULT_AGENT_COLOR 255,0,0
#define DEFAULT_OBSTACLE_REWARD 0
#define DEFAULT_MAP_MARGIN 3

// -- Prototypes --

class OAgent
{

public:

	cv::Mat stateMap;						// -- States
	cv::Mat visitMap;						// -- Visit map
	cv::Mat* qMap;							// --  Q(S,A)
	cv::Mat* nMap;							// --  N(S,-A)
	Vec2i agentPos;							// -- Agent Position (State)
	cv::Size mapSize;						// -- Domain size
	RewardGenerator* rewardEngine;			// -- Reward Enginge;
	QAgent* opponent;						// -- Agent Opponent
	// -- Distributions --

	std::uniform_int_distribution<int> rndAction;	// -- sampling from integer uniform  distribution for setting random states
	std::uniform_int_distribution<int> rndState;	// -- sampling from integer uniform  distribution for taking random action
	std::uniform_real_distribution<double> rndReal; // -- sampling from float uniform  distribution for other probability task
	RndEngine rndEngine;							// -- random variable generator engine


	// -- Rewards --

	float totalReward;
	Action lastAction;

	// -- Rates --

	float learningRate;
	float discountFactor;
	float greedyFactor;
	float rationalFactor;

	// -- Agent characteristics

	Vec3b agentColour;

	OAgent(cv::Mat& stateMap,bool randInit,
	RewardGenerator* rewardEngine,QAgent*opponent);

	Action getRandomAction();	// -- Get Random action

	bool takeGreedyAction();		// -- If agent should take a Greedy action

	bool takeRationalExplore();		// -- If agent should take a rational exploring action

	float takeAction(Action action);	// -- Take the specified action --

	Vec2i transFunction(Action action,bool& obstacle); // -- Transition function

	Action selectAction();

	float getBestAction(Vec2i position,Action& action);		// -- Search for the best action from the learned OMQ-Values

	float stepUpdate();	// -- Step forward and update Q-Map

	void reset(Vec2i position);	// -- Reset position - total reward and visit map


};



#endif /* QAGENT_HPP_ */
