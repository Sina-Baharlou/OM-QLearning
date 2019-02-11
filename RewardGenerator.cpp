/*
 * RewardGenerator.cpp
 *
 *  Created on: May 30, 2015
 *      Author: Sina M. Baharlou (Sina.baharlou@gmail.com)
 */


#include "RewardGenerator.hpp"




RewardGenerator::RewardGenerator(cv::Mat& stateMap,RndEngine rndEngine):stateMap(stateMap),rndEngine(rndEngine)
{
	goalMap=cv::Mat(stateMap.rows,stateMap.cols,CV_32FC1);		// -- Create goal map
	goalMap=0.0;

	rndReal=std::uniform_real_distribution<double>(0,1);
}

void RewardGenerator::addGoal(Vec2i pos,float value)		// -- Add goal to the map
{
	goalMap.at<float>(pos[1],pos[0])=value;
}

void RewardGenerator::addStochasticGoal(GaussSignal gaussSignal)
{
	stochasticGoals.push_back(gaussSignal);
}

float RewardGenerator::getReward(Vec2i agentPos)
{
	float reward=0;

	float probability=0;
	float rndFloat=0;

	// -- Get Stochastic reward --
	for (uint i=0;i<stochasticGoals.size();i++)
		probability+=normalProbability(stochasticGoals[i].mean,stochasticGoals[i].sigma,agentPos);

	rndFloat=rndReal(rndEngine);

	if (rndFloat<=probability)
		reward+=1;



	// -- Get deterministic reward --
	reward+=goalMap.at<float>(agentPos[1],agentPos[0]);

	return reward;
}

float RewardGenerator::getReward(Vec2i agentPos,Vec2i opponentPos)
{
	float reward=0;

	float probability=0;
	float rndFloat=0;

	// -- Get Stochastic reward --
	for (uint i=0;i<stochasticGoals.size();i++)
	{
		double pAgent=normalProbability(stochasticGoals[i].mean,stochasticGoals[i].sigma,agentPos);
		double pOpponent=normalProbability(stochasticGoals[i].mean,stochasticGoals[i].sigma,opponentPos);
		probability+=max(pAgent-pOpponent,0.0);

	}

	rndFloat=rndReal(rndEngine);

	if (rndFloat<=probability)
		reward+=1;



	// -- Get deterministic reward --
	reward+=goalMap.at<float>(agentPos[1],agentPos[0]);

	return reward;
}


float RewardGenerator::normalCDF(float mean,float sigma,float x)	// --  normal cumulative distribution function
{
	double temp= (x-mean)/ (   sqrt(sigma) *   sqrt(2));
	return 0.5 * ( 1+ erf( temp ));

}

float RewardGenerator::normalCDF2d (Vec2f mean,Vec2f sigma,Vec2f pos)		// -- Uncorrelated bivariate normal cumulative distribution function
{
	return normalCDF(mean(0),sigma(0),pos(0))*normalCDF(mean(1),sigma(1),pos(1));
}

float RewardGenerator::normalProbability(Vec2f mean,Vec2f sigma,Vec2i pos)
{
	float Z0=normalCDF2d(mean,sigma,Vec2f(pos(0)+1,pos(1)+1));
	float Z1=normalCDF2d(mean,sigma,Vec2f(pos(0)+1,pos(1)));
	float Z2=normalCDF2d(mean,sigma,Vec2f(pos(0),pos(1)+1));
	float Z3=normalCDF2d(mean,sigma,pos);
	return Z0-(Z1+Z2)+Z3;

}






