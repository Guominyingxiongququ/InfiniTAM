// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ITMDepthTracker.h"
#include "../../ORUtils/Cholesky.h"
//#include "../Objects/groundTruthPose.h"

#include <math.h>
#include <iostream>
using namespace ITMLib::Engine;

ITMDepthTracker::ITMDepthTracker(Vector2i imgSize, TrackerIterationType *trackingRegime, int noHierarchyLevels, int noICPRunTillLevel, float distThresh,
	float terminationThreshold, const ITMLowLevelEngine *lowLevelEngine, MemoryDeviceType memoryType)
{
    myfile.open("/home/xiyu/Desktop/project/InfiniTAM/InfiniTAM/Files/Teddy/kitti_pose.txt");
    if(myfile.is_open())
    {
    	std::cout<<"open file========================="<<std::endl;
    }
//	this->myPose = new GroundTruthPose("/home/xiyu/Desktop/project/InfiniTAM/InfiniTAM/Files/Teddy/kitti_pose.txt");
	viewHierarchy = new ITMImageHierarchy<ITMTemplatedHierarchyLevel<ITMFloatImage> >(imgSize, trackingRegime, noHierarchyLevels, memoryType, true);
	sceneHierarchy = new ITMImageHierarchy<ITMSceneHierarchyLevel>(imgSize, trackingRegime, noHierarchyLevels, memoryType, true);

	this->noIterationsPerLevel = new int[noHierarchyLevels];
	this->distThresh = new float[noHierarchyLevels];
	
	this->noIterationsPerLevel[0] = 2; //TODO -> make parameter
	for (int levelId = 1; levelId < noHierarchyLevels; levelId++)
	{
		noIterationsPerLevel[levelId] = noIterationsPerLevel[levelId - 1] + 2;
	}

	float distThreshStep = distThresh / noHierarchyLevels;
	this->distThresh[noHierarchyLevels - 1] = distThresh;
	for (int levelId = noHierarchyLevels - 2; levelId >= 0; levelId--)
		this->distThresh[levelId] = this->distThresh[levelId + 1] - distThreshStep;

	this->lowLevelEngine = lowLevelEngine;

	this->noICPLevel = noICPRunTillLevel;

	this->terminationThreshold = terminationThreshold;
}

ITMDepthTracker::~ITMDepthTracker(void) 
{ 
	delete this->viewHierarchy;
	delete this->sceneHierarchy;

	delete[] this->noIterationsPerLevel;
	delete[] this->distThresh;
	this->myfile.close();
}

void ITMDepthTracker::SetEvaluationData(ITMTrackingState *trackingState, const ITMView *view)
{
	this->trackingState = trackingState;
	this->view = view;

	sceneHierarchy->levels[0]->intrinsics = view->calib->intrinsics_d.projectionParamsSimple.all;
	viewHierarchy->levels[0]->intrinsics = view->calib->intrinsics_d.projectionParamsSimple.all;

	// the image hierarchy allows pointers to external data at level 0
	viewHierarchy->levels[0]->depth = view->depth;
	sceneHierarchy->levels[0]->pointsMap = trackingState->pointCloud->locations;
	sceneHierarchy->levels[0]->normalsMap = trackingState->pointCloud->colours;

	scenePose = trackingState->pose_pointCloud->GetM();
}

void ITMDepthTracker::PrepareForEvaluation()
{
	for (int i = 1; i < viewHierarchy->noLevels; i++)
	{
		ITMTemplatedHierarchyLevel<ITMFloatImage> *currentLevelView = viewHierarchy->levels[i], *previousLevelView = viewHierarchy->levels[i - 1];
		lowLevelEngine->FilterSubsampleWithHoles(currentLevelView->depth, previousLevelView->depth);
		currentLevelView->intrinsics = previousLevelView->intrinsics * 0.5f;

		ITMSceneHierarchyLevel *currentLevelScene = sceneHierarchy->levels[i], *previousLevelScene = sceneHierarchy->levels[i - 1];
		//lowLevelEngine->FilterSubsampleWithHoles(currentLevelScene->pointsMap, previousLevelScene->pointsMap);
		//lowLevelEngine->FilterSubsampleWithHoles(currentLevelScene->normalsMap, previousLevelScene->normalsMap);
		currentLevelScene->intrinsics = previousLevelScene->intrinsics * 0.5f;
	}
}

void ITMDepthTracker::SetEvaluationParams(int levelId)
{
	this->levelId = levelId;
	this->iterationType = viewHierarchy->levels[levelId]->iterationType;
	this->sceneHierarchyLevel = sceneHierarchy->levels[0];
	this->viewHierarchyLevel = viewHierarchy->levels[levelId];
}

void ITMDepthTracker::ComputeDelta(float *step, float *nabla, float *hessian, bool shortIteration) const
{
	for (int i = 0; i < 6; i++) step[i] = 0;

	if (shortIteration)
	{
		float smallHessian[3 * 3];
		for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) smallHessian[r + c * 3] = hessian[r + c * 6];

		ORUtils::Cholesky cholA(smallHessian, 3);
		cholA.Backsub(step, nabla);
	}
	else
	{
		ORUtils::Cholesky cholA(hessian, 6);
		cholA.Backsub(step, nabla);
	}
}

bool ITMDepthTracker::HasConverged(float *step) const
{
	float stepLength = 0.0f;
	for (int i = 0; i < 6; i++) stepLength += step[i] * step[i];

	if (sqrt(stepLength) / 6 < terminationThreshold) return true; //converged

	return false;
}

void ITMDepthTracker::ApplyDelta(const Matrix4f & para_old, const float *delta, Matrix4f & para_new) const
{
	float step[6];

	switch (iterationType)
	{
	case TRACKER_ITERATION_ROTATION:
		step[0] = (float)(delta[0]); step[1] = (float)(delta[1]); step[2] = (float)(delta[2]);
		step[3] = 0.0f; step[4] = 0.0f; step[5] = 0.0f;
		break;
	case TRACKER_ITERATION_TRANSLATION:
		step[0] = 0.0f; step[1] = 0.0f; step[2] = 0.0f;
		step[3] = (float)(delta[0]); step[4] = (float)(delta[1]); step[5] = (float)(delta[2]);
		break;
	default:
	case TRACKER_ITERATION_BOTH:
		step[0] = (float)(delta[0]); step[1] = (float)(delta[1]); step[2] = (float)(delta[2]);
		step[3] = (float)(delta[3]); step[4] = (float)(delta[4]); step[5] = (float)(delta[5]);
		break;
	}

	Matrix4f Tinc;

	Tinc.m00 = 1.0f;		Tinc.m10 = step[2];		Tinc.m20 = -step[1];	Tinc.m30 = step[3];
	Tinc.m01 = -step[2];	Tinc.m11 = 1.0f;		Tinc.m21 = step[0];		Tinc.m31 = step[4];
	Tinc.m02 = step[1];		Tinc.m12 = -step[0];	Tinc.m22 = 1.0f;		Tinc.m32 = step[5];
	Tinc.m03 = 0.0f;		Tinc.m13 = 0.0f;		Tinc.m23 = 0.0f;		Tinc.m33 = 1.0f;

	para_new = Tinc * para_old;
}

void ITMDepthTracker::TrackCamera(ITMTrackingState *trackingState, const ITMView *view)
{
	this->SetEvaluationData(trackingState, view);
	this->PrepareForEvaluation();

	float f_old = 1e10, f_new;
	int noValidPoints_new;

	float hessian_good[6 * 6], hessian_new[6 * 6], A[6 * 6];
	float nabla_good[6], nabla_new[6];
	float step[6];

	for (int levelId = viewHierarchy->noLevels - 1; levelId >= noICPLevel; levelId--)
	{
		this->SetEvaluationParams(levelId);
		if (iterationType == TRACKER_ITERATION_NONE) continue;

		Matrix4f approxInvPose = trackingState->pose_d->GetInvM();
		ITMPose lastKnownGoodPose(*(trackingState->pose_d));
		f_old = 1e20f;
		float lambda = 1.0;

		for (int iterNo = 0; iterNo < noIterationsPerLevel[levelId]; iterNo++)
		{
			// evaluate error function and gradients
			noValidPoints_new = this->ComputeGandH(f_new, nabla_new, hessian_new, approxInvPose);

			// check if error increased. If so, revert
			if ((noValidPoints_new <= 0)||(f_new > f_old)) {
				trackingState->pose_d->SetFrom(&lastKnownGoodPose);
				approxInvPose = trackingState->pose_d->GetInvM();
				lambda *= 10.0f;
			} else {
				lastKnownGoodPose.SetFrom(trackingState->pose_d);
				f_old = f_new;

				for (int i = 0; i < 6*6; ++i) hessian_good[i] = hessian_new[i] / noValidPoints_new;
				for (int i = 0; i < 6; ++i) nabla_good[i] = nabla_new[i] / noValidPoints_new;
				lambda /= 10.0f;
			}
			for (int i = 0; i < 6*6; ++i) A[i] = hessian_good[i];
			for (int i = 0; i < 6; ++i) A[i+i*6] *= 1.0f + lambda;

			// compute a new step and make sure we've got an SE3
			ComputeDelta(step, nabla_good, A, iterationType != TRACKER_ITERATION_BOTH);
			ApplyDelta(approxInvPose, step, approxInvPose);
			trackingState->pose_d->SetInvM(approxInvPose);
			trackingState->pose_d->Coerce();
			approxInvPose = trackingState->pose_d->GetInvM();

			// if step is small, assume it's going to decrease the error and finish
			if (HasConverged(step)) break;
		}
	}
/* ground truth
	std::cout<<"Tracking Pose"<<std::endl<<trackingState->pose_d->GetInvM()<<std::endl;
	double poseValue[3][4];
	Matrix4f currentPose = * (new Matrix4f());
	Matrix4f kinectTrans = Matrix4f();
	float kinectTransVal[16] = {1.0, 0.0 , 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
	for(int i=0;i<16;i++)
	{
		kinectTrans.m[i] = kinectTransVal[i];
	}
	Matrix4f kinectTransInv = Matrix4f();
	kinectTrans.inv(kinectTransInv);
	std::cout<<"inverse kinect pose"<<kinectTransInv<<std::endl;
	Matrix4f transPose = Matrix4f();
	Matrix4f invPose = Matrix4f();
	Matrix4f inversePose = Matrix4f();
	Matrix4f transInvPose = Matrix4f();
	std::cout<<"GroundTruth Tracking Pose"<<std::endl;
	for(int i=0;i<3;i++)
	{
		for(int j=0;j<4;j++)
		{
			myfile>>poseValue[i][j];
			currentPose.m[i*4+j]= poseValue[i][j];
			std::cout<<poseValue[i][j]<<" ";
		}
		std::cout<<std::endl;
	}
	currentPose.m[12] = 0.0;
	currentPose.m[13] = 0.0;
	currentPose.m[14] = 0.0;
	currentPose.m[15] = 1.0;

	transPose = kinectTrans* currentPose * kinectTransInv;


//	for(int i=0;i<3;i++)
//	{
//		for(int j=0;j<3;j++)
//		{
//			inversePose.m[4*j+i] = transPose.m[4*i+j];
//		}
//	}
//
//	for(int i=0;i<3;i++)
//	{
//		inversePose.m[4*i+3] = 0.0;
//		for(int j=0;j<3;j++)
//		{
//			inversePose.m[4*i+3] = inversePose.m[4*i+3] - transPose.m[4*i+3]*inversePose.m[4*i+j];
//		}
//	}
//	inversePose.m[12] = 0.0;
//	inversePose.m[13] = 0.0;
//	inversePose.m[14] = 0.0;
//	inversePose.m[15] = 1.0;
//

	transPose.inv(invPose);
	std::cout<<invPose<<std::endl;
	for(int i=0;i<4;i++)
	{
		for(int j=0;j<4;j++)
		{
			transInvPose.m[4*i+j] = invPose.m[4*j+i];
		}
	}
	trackingState->pose_d->SetM(transInvPose);
	*/
	trackingState->pose_d->SetM(trackingState->pose_d->GetM());
    std::cout<<"Tracking Pose"<<std::endl<<trackingState->pose_d->GetM()<<std::endl;
//	Matrix4f * currentPose = &(this->myPose.setPose());
//	trackingState->pose_d->SetM(* currentPose);
}

