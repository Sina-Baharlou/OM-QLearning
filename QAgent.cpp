/*
 * QAgent.cpp
 *
 *  Created on: May 30, 2015
 *      Author: Sina M. Baharlou (Sina.baharlou@gmail.com)
 */


#include "QAgent.hpp"

	QAgent::QAgent(cv::Mat& stateMap,bool randInit,	RewardGenerator* rewardEngine):
		stateMap(stateMap),
		rewardEngine(rewardEngine),
		opponentPos(NULL)
	{

		mapSize=cv::Size(stateMap.cols,stateMap.rows);

		// -- Create Visit Map
		visitMap=cv::Mat(stateMap.rows,stateMap.cols,CV_8UC1);
		visitMap=(cv::Scalar)0;

		// -- Create Q-Map matrices
		qMap=new cv::Mat[ACTION_COUNT];
		for (int i=FIRST_ACTION;i<=LAST_ACTION;i++)
		{
			qMap[i]=cv::Mat(mapSize,CV_32FC1);	// -- Create 32Bit float matrix
			qMap[i]=0.0;						// -- Set initial zero value
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



	Action QAgent::getRandomAction()	// -- Get Random action
	{
		return (Action)rndAction(rndEngine);
	}


	bool QAgent::takeGreedyAction()		// -- If agent should take a Greedy action
	{

		double random=rndReal(rndEngine);	// -- Get a double random number between 0 and 1

		// -- Check if agent should take a greedy action
		if (random<=greedyFactor)
			return true;
		return false;
	}

	bool QAgent::takeRationalExplore()		// -- If agent should take a rational exploring action
	{

		double random=rndReal(rndEngine);	// -- Get a double random number between 0 and 1

		// -- Check if agent should take a greedy action
		if (random<=rationalFactor)
			return true;
		return false;
	}

	float QAgent::takeAction(Action action)	// -- Take the specified action --
	{
		bool obstacle=false;
		float reward=0;

		agentPos=transFunction(action,obstacle); 		// -- Update the position

		// -- Get the reward --
		if (obstacle)
			reward=DEFAULT_OBSTACLE_REWARD;
		else
		{
			if (opponentPos!=NULL)
				reward=rewardEngine->getReward(agentPos,*opponentPos); // -- Take the reward from Reward engine
			else
				reward=rewardEngine->getReward(agentPos);
		}
		return reward;
	}

	Vec2i QAgent::transFunction(Action action,bool& obstacle) // -- Transition function
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

	Action QAgent::selectAction()
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
						int visitCount=visitMap.at<unsigned char>(newPos(1),newPos(0));
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

	float QAgent::getBestAction(Vec2i position,Action& action)		// -- Search for the best action from the learned Q-Values
	{
		action=RIGHT;

		float qMax=-std::numeric_limits<float>::min();

		float qTemp;

		// -- Search
		for (int i=FIRST_ACTION;i<=LAST_ACTION;i++)
		{
			qTemp=qMap[i].at<float>(position[1],position[0]);
			if ( qTemp>qMax)
			{
				qMax=qTemp;
				action=(Action)i;
			}
		}

		return qMax;

	}

	float QAgent::stepUpdate()	// -- Step forward and update Q-Map
	{

		float reward;
		Vec2i oldPos=agentPos;


		// -- Select an action--
		Action action=selectAction();

		// -- Take the action and get the reward --
		reward=takeAction(action);
		totalReward+=reward;

		// -- Find maximum Q --
		Action act;
		float qMax=getBestAction(agentPos,act);

		// -- Get Current Q Value
		float currentQ=qMap[action].at<float>(oldPos[1],oldPos[0]);

		// -- Update QMap
		qMap[action].at<float>(oldPos[1],oldPos[0])=currentQ+ learningRate * (reward+discountFactor*qMax-currentQ);

		// -- Update visit map
		visitMap.at<unsigned char>(oldPos(1),oldPos(0))++;	// -- Mark as visited

		return reward;
	}

	void QAgent::reset(Vec2i position)	// -- Reset position - total reward and visit map
	{
		agentPos=position;
		totalReward=0;
		visitMap=(Scalar)0.0;
	}










