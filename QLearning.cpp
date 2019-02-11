//============================================================================
// Name        : QLearning.cpp
// Author      : Sina M.Baharlou
// Version     :
// Copyright   :
// Description :
//============================================================================

// -- Includes --
#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <random>
#include <limits>

// -- OpenCV --
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


// -- My headers --
#include "Actions.hpp"
#include "RewardGenerator.hpp"
#include "QAgent.hpp"			// -- Q-learning agent
#include "OAgent.hpp"			// -- OMQ-learning agent

using namespace std;
using namespace cv;


// -- Definitions --
#define MAP_FILE "map.png"
#define GAUSS1_POS	5,5
#define GAUSS2_POS 	15,15
#define DEFAULT_COV  1,1
#define EPISODE_LENGTH 100
#define TOTAL_ITERATIONS 500
#define OPPONENT

// -- Global variables --
std::uniform_int_distribution<int> rndState;
RndEngine generator;

// -- Global methods --

cv::Mat enlargeMat(cv::Mat input,float scaleFactor,int interpolation=CV_INTER_NN) 
{
	cv::Mat output;
	cv::Size size(input.rows*scaleFactor,input.cols*scaleFactor);

	cv::resize(input,output,size,0,0,interpolation);
	return output;
}


void makePolicyMap(cv::Mat& stateMap,QAgent* agent)
{

	cv::Mat up=imread("arrows/U.png");
	cv::cvtColor( up, up, CV_BGR2GRAY );	// -- Convert map to grayscale --

	cv::Mat down=imread("arrows/D.png");
	cv::cvtColor( down, down, CV_BGR2GRAY );	// -- Convert map to grayscale --


	cv::Mat right=imread("arrows/R.png");
	cv::cvtColor( right, right, CV_BGR2GRAY );	// -- Convert map to grayscale --

	cv::Mat left=imread("arrows/L.png");
	cv::cvtColor( left, left, CV_BGR2GRAY );	// -- Convert map to grayscale --

	int width=up.cols;
	int height=up.rows;

	cv::Mat policy(stateMap.rows *height ,stateMap.cols*width,CV_8UC1);
	policy=(Scalar)255;

	Action a=RIGHT;
	for (int i=0;i<stateMap.rows;i++)
		for(int j=0;j<stateMap.cols;j++)
		{
			float value=agent->getBestAction(Vec2i(j,i),a);

			Mat arrow;

			switch (a)
			{
				case UP: 	arrow=up;	break;
				case DOWN:	arrow=down;	break;
				case LEFT:	arrow=left;	break;
				case RIGHT:	arrow=right;break;
				case STAY:	arrow=up;	break;
			}


			cv::Rect roi( cv::Point( j*width, i*height ), arrow.size() );

			cv::Mat temp;
			arrow.copyTo(temp);

			temp=-(temp-255) *(value*0.9+0.2);
			temp=255-temp;
			temp.copyTo( policy( roi ) );

		}

	imwrite("policyQ.png",policy);

}



void makePolicyMap(cv::Mat& stateMap,OAgent* agent)
{

	cv::Mat up=imread("arrows/U.png");
	cv::cvtColor( up, up, CV_BGR2GRAY );	// -- Convert map to grayscale --

	cv::Mat down=imread("arrows/D.png");
	cv::cvtColor( down, down, CV_BGR2GRAY );	// -- Convert map to grayscale --


	cv::Mat right=imread("arrows/R.png");
	cv::cvtColor( right, right, CV_BGR2GRAY );	// -- Convert map to grayscale --

	cv::Mat left=imread("arrows/L.png");
	cv::cvtColor( left, left, CV_BGR2GRAY );	// -- Convert map to grayscale --

	int width=up.cols;
	int height=up.rows;

	cv::Mat policy(stateMap.rows *height ,stateMap.cols*width,CV_8UC1);
	policy=(Scalar)255;

	Action a=RIGHT;
    
	for (int i=0;i<stateMap.rows;i++)
		for(int j=0;j<stateMap.cols;j++)
		{
			float value=agent->getBestAction(Vec2i(j,i),a);

			Mat arrow;

			switch (a)
			{
				case UP: 	arrow=up;	break;
				case DOWN:	arrow=down;	break;
				case LEFT:	arrow=left;	break;
				case RIGHT:	arrow=right;break;
				case STAY:	arrow=up;	break;
			}

			cv::Rect roi( cv::Point( j*width, i*height ), arrow.size() );
			cv::Mat temp;
			arrow.copyTo(temp);

			temp=-(temp-255) *(value*0.9+0.2);
			temp=255-temp;
			temp.copyTo( policy( roi ) );

		}


	imwrite("policyO.png",policy);

}


cv::Mat normMat(cv::Mat& input)
{
	double min;
	double max;
	cv::minMaxIdx(input, &min, &max);
	cv::Mat adjMap;
	cv::convertScaleAbs(input, adjMap, 255 / max);
	return adjMap;
}


int main(int argc,char* argv[])
{


	cv::namedWindow("Q-State Map");
	cv::namedWindow("Q-Visited Map");
	cv::namedWindow("Q-Value Function Map");
	cv::namedWindow("Q-Best Policy Map");

#ifdef OPPONENT
	cv::namedWindow("O-Visited Map");
	cv::namedWindow("O-Value Function Map");
	cv::namedWindow("O-Best Policy Map");
#endif

	// -- Load default map --
	cv::Mat stateMap=imread(MAP_FILE,1);
	cv::cvtColor( stateMap, stateMap, CV_BGR2GRAY );	// -- Convert map to grayscale --

	// -- Create a matrix for final rendering --
	cv::Mat finalRender;

	// -- Init Reward class
	RewardGenerator* rwdGen=new RewardGenerator(stateMap,generator);

	// -- Add two gaussian distribution stochastic goal
	rwdGen->addStochasticGoal(GaussSignal(Vec2f(GAUSS1_POS),Vec2f(DEFAULT_COV),0.1));
	rwdGen->addStochasticGoal(GaussSignal(Vec2f(GAUSS2_POS),Vec2f(DEFAULT_COV),0.1));

	rndState=std::uniform_int_distribution<int>(3,stateMap.rows-3);


	// -- Initializing the learners --

	QAgent qAgent(stateMap,true,rwdGen);	// -- Create a Q-learner agent

#ifdef OPPONENT
	OAgent oAgent(stateMap,false,rwdGen,&qAgent);	// -- Create q OMQ-Learner agent
	oAgent.agentColour=Vec3b(0,0,255);
	qAgent.opponentPos=&oAgent.agentPos;
#endif
	// -- Seed the random generator
	generator.seed(rand());


	// -- Create two matrices for showing best q-values --
	cv::Mat qBestValue=cv::Mat(stateMap.rows,stateMap.cols,CV_32FC1);
	qBestValue=255.0;


	cv::Mat qBestPolicy=cv::Mat(stateMap.rows,stateMap.cols,CV_8UC3);
	qBestPolicy=(Scalar)255.0;

#ifdef OPPONENT
	cv::Mat oBestValue=cv::Mat(stateMap.rows,stateMap.cols,CV_32FC1);
	oBestValue=255.0;

	cv::Mat oBestPolicy=cv::Mat(stateMap.rows,stateMap.cols,CV_8UC3);
	oBestPolicy=(Scalar)255.0;
#endif

	int timestep=0;
	int episode=0;



	while(true)
	{
		// -- Copy state map to finalRender Map
		stateMap.copyTo(finalRender);
		cv::cvtColor( finalRender, finalRender, CV_GRAY2BGR ); // -- Convert it back to RGB

		// -- Draw agents --
		finalRender.at<cv::Vec3b>(qAgent.agentPos(1),qAgent.agentPos(0))=qAgent.agentColour;
#ifdef OPPONENT
		finalRender.at<cv::Vec3b>(oAgent.agentPos(1),oAgent.agentPos(0))=oAgent.agentColour;
#endif

		// -- Update agents
		qAgent.stepUpdate();
#ifdef OPPONENT
		oAgent.stepUpdate();
#endif
		timestep++;

		//	-- Reach end of the episode
		if (timestep==EPISODE_LENGTH)
		{
			timestep=0;
			episode++;

			cout<<qAgent.totalReward<<" ";
			qAgent.reset(Vec2i(4,18));
			qAgent.greedyFactor+=0.002;
            
#ifdef OPPONENT
			cout<<oAgent.totalReward<<";"<<endl;
			oAgent.reset(Vec2i(4,18));
			oAgent.greedyFactor+=0.002;
#endif
				cv::waitKey(3);			// -- Wait 

		}
		if (episode==TOTAL_ITERATIONS)break;

		Action a=RIGHT;
		for (int i=0;i<stateMap.rows;i++)
			for(int j=0;j<stateMap.cols;j++)
			{
				qBestValue.at<float>(i,j)=1-qAgent.getBestAction(Vec2i(j,i),a);
				float mag=qAgent.getBestAction(Vec2i(j,i),a);

				Vec3b color;
				switch (a)
				{
				case UP: 	color=Vec3b(200,200,0);	break;
							case DOWN:	color=Vec3b(200,0,200);	break;
							case LEFT:	color=Vec3b(0,200,200);	break;
							case RIGHT:	color=Vec3b(200,0,0);	break;
							case STAY:	color=Vec3b(0,0,0);	break;
				}
				qBestPolicy.at<Vec3b>(i,j)=Vec3b(255,255,255)-(color*mag*10);
			}




		//--
#ifdef OPPONENT
		for (int i=0;i<stateMap.rows;i++)
			for(int j=0;j<stateMap.cols;j++)
			{
				oBestValue.at<float>(i,j)=1-oAgent.getBestAction(Vec2i(j,i),a);
				float mag=oAgent.getBestAction(Vec2i(j,i),a);
				Vec3b color;
				switch (a)
				{
				case UP: 	color=Vec3b(200,200,0);	break;
				case DOWN:	color=Vec3b(200,0,200);	break;
				case LEFT:	color=Vec3b(0,200,200);	break;
				case RIGHT:	color=Vec3b(200,0,0);	break;
				case STAY:	color=Vec3b(0,0,0);	break;
				}
				oBestPolicy.at<Vec3b>(i,j)=Vec3b(255,255,255)-(color*mag*10);

			}
#endif

		//--

        cv::Mat enlarged=enlargeMat(finalRender,12,CV_INTER_CUBIC);
		circle(enlarged,Point(GAUSS1_POS)*13,10,Scalar(0,255,0),2,CV_AA );
		circle(enlarged,Point(GAUSS2_POS)*13,10,Scalar(0,255,0),2,CV_AA );


		finalRender.at<cv::Vec3b>(qAgent.agentPos(1),qAgent.agentPos(0))=qAgent.agentColour;
		cv::imshow("Q-State Map",enlarged);

		cv::imshow("Q-Visited Map",enlargeMat(qAgent.visitMap*10,12,CV_INTER_CUBIC));
		cv::imshow("Q-Value Function Map",enlargeMat(normMat(qBestValue),12,CV_INTER_CUBIC));
		cv::imshow("Q-Best Policy Map",enlargeMat(qBestPolicy,12,CV_INTER_CUBIC));


#ifdef OPPONENT

		cv::imshow("O-Value Function Map",enlargeMat(normMat(oBestValue),12,CV_INTER_CUBIC));
		cv::imshow("O-Best Policy Map",enlargeMat(oBestPolicy,12,CV_INTER_CUBIC));
#endif


	}

	// ** Uncomment the following lines to generate further results ** //
	/*
	cv::imwrite("BestV.png",enlargeMat(normMat(qBestValue),12,CV_INTER_CUBIC));
	cv::imwrite("BestP.png",enlargeMat(normMat(qBestPolicy),12,CV_INTER_CUBIC));


	cv::imwrite("BestVO.png",enlargeMat(normMat(oBestValue),12,CV_INTER_CUBIC));
	cv::imwrite("BestPO.png",enlargeMat(normMat(oBestPolicy),12,CV_INTER_CUBIC));

	makePolicyMap(stateMap,&qAgent);
	*/

	return 0;
}

