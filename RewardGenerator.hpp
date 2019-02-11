/*
 * RewardGenerator.hpp
 *
 *  Created on: May 26, 2015
 *      Author: Sina M. Baharlou (Sina.baharlou@gmail.com)
 */

#ifndef REWARDGENERATOR_HPP_
#define REWARDGENERATOR_HPP_



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

typedef std::default_random_engine RndEngine;
using namespace std;
using namespace cv;


// -- Prototypes --

struct GaussSignal
{

	Vec2f mean;
	Vec2f sigma;
	float magnitude;
	GaussSignal(Vec2f mean,Vec2f sigma,float magnitude):mean(mean),sigma(sigma),magnitude(magnitude){}
};

class RewardGenerator
{
public :
	cv::Mat stateMap;	// -- States
	cv::Mat goalMap;	// -- Deterministic Goals
	std::vector< GaussSignal > stochasticGoals; // -- Stochastic Goals
	std::uniform_real_distribution<double> rndReal;
	RndEngine rndEngine;// -- random variable generator engine

	RewardGenerator(cv::Mat& stateMap,RndEngine rndEngine);

	void addGoal(Vec2i pos,float value);		// -- Add goal to the map

	void addStochasticGoal(GaussSignal gaussSignal);

	float getReward(Vec2i agentPos);

	float getReward(Vec2i agentPos,Vec2i opponentPos);

	float normalCDF(float mean,float sigma,float x);	// --  normal cumulative distribution function

	float normalCDF2d (Vec2f mean,Vec2f sigma,Vec2f pos);		// -- Uncorrelated bivariate normal cumulative distribution function

	float normalProbability(Vec2f mean,Vec2f sigma,Vec2i pos);



};

#endif /* REWARDGENERATOR_HPP_ */
