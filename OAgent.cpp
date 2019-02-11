/*
 * OAgent.cpp
 *
 *  Created on: May 30, 2015
 *      Author: Sina M. Baharlou (Sina.baharlou@gmail.com)
 */

#include "OAgent.hpp"



OAgent::OAgent(cv::Mat& stateMap,bool randInit,
		RewardGenerator* rewardEngine,QAgent*opponent):
		stateMap(stateMap),
		rewardEngine(rewardEngine),
		opponent(opponent)
{

	mapSize=cv::Size(stateMap.cols,stateMap.rows);

	// -- Create Visit Map
	visitMap=cv::Mat(stateMap.rows,stateMap.cols,CV_32SC1);
	visitMap=(cv::Scalar)0;

	// -- Create Q-Map matrices
	qMap=new cv::Mat[ACTION_COUNT*ACTION_COUNT];

	for (int i=FIRST_ACTION;i<=LAST_ACTION;i++)
	{
		for (int j=FIRST_ACTION;j<=LAST_ACTION;j++)
		{
			qMap[i*ACTION_COUNT +j]=cv::Mat(stateMap.rows,stateMap.cols,CV_32FC1);	// -- Create 32Bit float matrix
			qMap[i*ACTION_COUNT +j]=0.0;						// -- Set initial zero value
		}

	}

	// -- Create N-Map matrices
	nMap=new cv::Mat[ACTION_COUNT];

	for (int i=FIRST_ACTION;i<=LAST_ACTION;i++)
	{
		nMap[i]=cv::Mat(stateMap.rows,stateMap.cols,CV_32SC1);	// -- Create 32Bit float matrix
		nMap[i]=(cv::Scalar)0;				// -- Set initial zero value
	}

	// -- Initializing distribution variables --

	rndAction=std::uniform_int_distribution<int>(FIRST_ACTION,LAST_ACTION);
	rndState=std::uniform_int_distribution<int>(DEFAULT_MAP_MARGIN,MAX(mapSize.width,mapSize.height)-DEFAULT_MAP_MARGIN);
	rndReal=std::uniform_real_distribution<double>(0,1);

	// -- Init random engine --

	 struct timespec ts;
	 unsigned theTick = 0U;
	 clock_gettime( CLOCK_REALTIME, &ts );
	 theTick  = ts.tv_nsec / 1000000;
	 theTick += ts.tv_sec * 1000;
	 rndEngine.seed(theTick*rand());

	//-- Initialize agent state

	if (randInit)
		agentPos=Vec2i(rndState(rndEngine),rndState(rndEngine));
	else
		agentPos=Vec2i(DEFAULT_POS);

	visitMap.at<unsigned char>(agentPos(1),agentPos(0))++;

	// -- Initialize reward variables

	totalReward=0.0;
	lastAction=RIGHT;

	// -- Init rates --

	learningRate=DEFAULT_LR;
	discountFactor=DEFAULT_DF;
	greedyFactor=DEFAULT_GR;
	rationalFactor=DEFAULT_RF;
	// -- Agent characteristics

	agentColour=Vec3b(DEFAULT_AGENT_COLOR); //
}



Action OAgent::getRandomAction()	// -- Get Random action
{
	return (Action)rndAction(rndEngine);
}


bool OAgent::takeGreedyAction()		// -- If agent should take a Greedy action
{

	double random=rndReal(rndEngine);	// -- Get a double random number between 0 and 1

	// -- Check if agent should take a greedy action
	if (random<=greedyFactor)
		return true;
	return false;
}

bool OAgent::takeRationalExplore()		// -- If agent should take a rational exploring action
{

	double random=rndReal(rndEngine);	// -- Get a double random number between 0 and 1

	// -- Check if agent should take a greedy action
	if (random<=rationalFactor)
		return true;
	return false;
}

float OAgent::takeAction(Action action)	// -- Take the specified action --
{
	bool obstacle=false;
	float reward=0;

	agentPos=transFunction(action,obstacle); 		// -- Update the position

	// -- Get the reward --
	if (obstacle)
		reward=DEFAULT_OBSTACLE_REWARD;
	else
		reward=rewardEngine->getReward(agentPos,opponent->agentPos); // -- Take the reward from Reward engine

	return reward;
}

Vec2i OAgent::transFunction(Action action,bool& obstacle) // -- Transition function
{

	obstacle=false;
	int agentX=agentPos[0];
	int agentY=agentPos[1];

	// -- Apply the action --
	switch (action)
	{

	case UP: 	agentY--;	break;
	case DOWN:	agentY++;	break;
	case LEFT:	agentX--;	break;
	case RIGHT:	agentX++;	break;
	case STAY:				break;
	}

	// -- Check boundaries and obstacles --
	if (agentX<0 || agentY<0 ||
			agentX>mapSize.width ||
			agentY>mapSize.height ||
			stateMap.at<uchar>(agentY,agentX)==0)
	{
		obstacle=true;
		return agentPos;
	}

	return Vec2i(agentX,agentY);

}

Action OAgent::selectAction()
{
	Action action;


	// -- Take random action --

	if (!takeGreedyAction())
	{

		if (takeRationalExplore())			// -- Take rational random action (Explore the least visited nodes )
		{
			bool obs=false;
			int leastVisited=std::numeric_limits<int>::max();


			// -- Search for the least visited node --

			for (int i=FIRST_ACTION;i<=LAST_ACTION;i++)
			{
				Vec2i newPos=transFunction((Action)i,obs);
				if (obs)
					continue;
				int visitCount=visitMap.at<int>(newPos(1),newPos(0));
				if (visitCount<leastVisited)
				{
					leastVisited=visitCount;
					action=(Action)i;
				}
			}

			// --

		}
		else
		{
			action=getRandomAction();
		}

		return action;
	}


	// -- Take Greedy action --

	getBestAction(agentPos,action);

	return action;

}

float OAgent::getBestAction(Vec2i position,Action& action)		// -- Search for the best action from the learned OMQ-Values
{
	action=RIGHT;
	int nS= visitMap.at<int>(position[1],position[0]);
	int cS= 0;
	float totalSum=0;
	float maxSum=-std::numeric_limits<float>::min();

	for (int i=FIRST_ACTION;i<=LAST_ACTION;i++)
	{
		totalSum=0;

		for (int j=FIRST_ACTION;j<=LAST_ACTION;j++)
		{
			cS=nMap[j].at<int>(position[1],position[0]);
			totalSum+=((float)cS/nS)*qMap[i*ACTION_COUNT+j].at<float>(position[1],position[0]);
		}

		if (totalSum>maxSum) { maxSum=totalSum; action=(Action)i; }

	}
	// ---
	return maxSum;

}

float OAgent::stepUpdate()	// -- Step forward and update Q-Map
{

	float reward;
	Vec2i oldPos=agentPos;


	// -- Select an action--
	Action action=selectAction();

	// -- Take the action and get the reward --
	reward=takeAction(action);
	totalReward+=reward;

	// -- Observer oponent action

	Action oAction=opponent->lastAction;

	// -- Find maximum Q --
	Action act;
	float vMax=getBestAction(agentPos,act);

	// -- Get Current Q Value
	float currentQ=qMap[action*ACTION_COUNT+oAction].at<float>(oldPos[1],oldPos[0]);

	// -- Update QMap
	qMap[action*ACTION_COUNT+oAction].at<float>(oldPos[1],oldPos[0])=currentQ+ learningRate * (reward+discountFactor*vMax-currentQ);

	// -- Update Visit map
	visitMap.at<int>(oldPos[1],oldPos[0])++;	// -- Mark as visited

	// -- Update N-Map

	nMap[oAction].at<int>(oldPos[1],oldPos[0])++;


	return reward;
}

void OAgent::reset(Vec2i position)	// -- Reset position - total reward and visit map
{
	agentPos=position;
	totalReward=0;
	//visitMap=(Scalar)0.0;
}




