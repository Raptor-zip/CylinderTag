#include "header/CylinderTag.h"

using namespace std;
using namespace cv;

CylinderTag::CylinderTag(const string& path)
{
	load_from_file(path);
}

CylinderTag::CylinderTag(const Mat1i& set_state)
{
	load_from_set(set_state);
}

void CylinderTag::load_from_file(const string path)
{
	ifstream input_file(path);
	if (!input_file.is_open())
	{
		throw __FUNCTION__ + string(", ") + "could not open the file\n";
	}
	
	int marker_num, marker_col, feature_size;
	input_file >> marker_num >> marker_col >> feature_size;
	
	this->featureSize = feature_size;
	this->state = Mat1i(marker_num, marker_col);
	for (int& i : this->state)
	{
		input_file >> i;
	}
	try
	{
		check_dictionary(this->state);
	}
	catch (const string s)
	{
		throw s + __FUNCTION__ + string(", ") + "illegal marker info\n";
	}
}

void CylinderTag::load_from_set(const Mat1i& set_state)
{
	try
	{
		check_dictionary(set_state);
	}
	catch (const string s)
	{
		throw s + __FUNCTION__ + string(", ") + "illegal marker info\n";
	}
	this->state = set_state;
}

void CylinderTag::check_dictionary(const Mat1i& input_state)
{
		for (int i : input_state)
		{
			if (!(i >= 0 && i <= 63))
				throw __FUNCTION__ + string(", ") + "the number in state matrix must between 0 to 63\n";
		}
		
		return;
}

void CylinderTag::detect(const Mat& img, vector<MarkerInfo>& result, int adaptiveThresh, const bool refine, int radius) {
    if(img.empty()||img.type()!=CV_8UC1||adaptiveThresh<1||radius<1)
        throw std::invalid_argument("detect expects a nonempty 8-bit grayscale image and positive windows");
    vector<int> windows={adaptiveThresh};
    if(multiThreshold) windows={adaptiveThresh,2*adaptiveThresh+1,4*adaptiveThresh+1};
    vector<MarkerInfo> hypotheses;
    DetectionStats selected;
    vector<MarkerInfo> selectedCandidates;
    vector<pair<string,Mat>> selectedImages;
    double bestStrength=-1e9;
    for(int window : windows) {
        vector<MarkerInfo> found;
        detectSingle(img,found,window,refine,radius);
        double strength=stats.features;
        for(const auto& marker : found)
            strength=std::max(strength,100.0*marker.supportColumns-marker.decodeError);
        if(strength>bestStrength) {
            bestStrength=strength; selected=stats;
            selectedCandidates=candidateMarkers; selectedImages=stageImages;
        }
        hypotheses.insert(hypotheses.end(),found.begin(),found.end());
    }
    // Keep the strongest spatial hypothesis across threshold windows, without
    // giving any preference to a particular ID (including the test strip ID).
    std::stable_sort(hypotheses.begin(),hypotheses.end(),[](const auto& a,const auto& b){
        if(a.supportColumns!=b.supportColumns) return a.supportColumns>b.supportColumns;
        return a.decodeError<b.decodeError;
    });
    auto bounds=[](const MarkerInfo& marker) {
        vector<Point2f> points;
        for(const auto& column : marker.cornerLists) points.insert(points.end(),column.begin(),column.end());
        return boundingRect(points);
    };
    result.clear();
    for(const auto& candidate : hypotheses) {
        const Rect box=bounds(candidate);
        bool duplicate=false;
        for(const auto& accepted : result) {
            const Rect other=bounds(accepted);
            const double overlap=(box&other).area();
            if(overlap>.3*std::min(box.area(),other.area())) {duplicate=true;break;}
        }
        if(!duplicate) result.push_back(candidate);
    }
    stats=selected;candidateMarkers=std::move(selectedCandidates);stageImages=std::move(selectedImages);
    if(debugCapture) {
        Mat overlay; cvtColor(img,overlay,COLOR_GRAY2BGR);
        for(const auto& marker : result) {
            const auto color=Scalar(20,220,20);
            for(const auto& column : marker.cornerLists) for(int half=0;half<2;++half) {
                vector<Point> p(column.begin()+half*4,column.begin()+half*4+4);
                polylines(overlay,p,true,color,2,LINE_AA);
            }
            const Rect box=bounds(marker);
            putText(overlay,"ID "+std::to_string(marker.markerID)+(marker.mirrored?" MIRROR":""),
                {box.x,std::max(15,box.y-7)},FONT_HERSHEY_SIMPLEX,.5,color,2,LINE_AA);
        }
        stageImages.push_back({"06_decoded",overlay});
    }
}

void CylinderTag::detectSingle(const Mat& img, vector<MarkerInfo>& markers_info, int adaptiveThresh, const bool cornerSubPix, int cornerSubPixDist){
    stats = DetectionStats{};
    stats.selectedWindow=adaptiveThresh;
    stageImages.clear();
    if(debugCapture) stageImages.push_back({"01_gray",img.clone()});
    candidateMarkers.clear();
    markers_info.clear();

	//Refresh
	quadAreas_labeled.clear();
	corners.clear();
	features.clear();
	markers.clear();

	Mat img_resize, img_float;
    img.convertTo(img_resize, CV_32FC1, 1.0 / 255);

    Mat img_binary(img_resize.rows, img_resize.cols, CV_8UC1);
    detector.adaptiveThreshold(img_resize, img_binary, adaptiveThresh);
    if(debugCapture) stageImages.push_back({"02_binary",img_binary.clone()});
	detector.connectedComponentLabeling(img_binary, quadAreas_labeled);
    stats.components = quadAreas_labeled.size();

    detector.contourQuads(img_binary, corners);
    if (cornerSubPix) detector.refineQuads(img_resize, corners, cornerSubPixDist);
    stats.quads = corners.size();
    if(debugCapture) {
        Mat overlay; cvtColor(img,overlay,COLOR_GRAY2BGR);
        for(const auto& quad : corners) {
            vector<Point> points(quad.begin(),quad.end());
            polylines(overlay,points,true,Scalar(20,220,20),1,LINE_AA);
        }
        stageImages.push_back({"03_quads",overlay});
    }
	if (corners.empty()) {
		return;
	}

	detector.featureRecovery(corners, features);
    for (auto& feature : features) {
        auto& p=feature.corners;
        const Point2f longitudinal=(p[0]+p[1]-p[4]-p[5])/2;
        const bool vertical=std::abs(longitudinal.y)>=std::abs(longitudinal.x);
        const auto along=[&](Point2f q){ return vertical?q.y:q.x; };
        const auto across=[&](Point2f q){ return vertical?q.x:q.y; };
        // Contour enumeration is not a top-to-bottom scan. Canonicalize the
        // two halves before decoding instead of depending on component order.
        if(along(p[0]+p[1])>along(p[4]+p[5]))
            for(int k=0;k<4;++k) std::swap(p[k],p[k+4]);
        if(across(p[0])>across(p[1])) {
            std::swap(p[0],p[1]); std::swap(p[2],p[3]);
            std::swap(p[4],p[5]); std::swap(p[6],p[7]);
        }
        feature.feature_center=(p[0]+p[1]+p[4]+p[5])/4;
        feature.feature_angle=std::atan2((p[0]+p[1]-p[4]-p[5]).y,
                                        (p[0]+p[1]-p[4]-p[5]).x)*180/CV_PI;
    }
    stats.features = features.size();
    if(debugCapture) {
        Mat overlay; cvtColor(img,overlay,COLOR_GRAY2BGR);
        for(size_t i=0;i<features.size();++i) {
            const auto& f=features[i];
            for(int half=0;half<2;++half) {
                vector<Point> points(f.corners.begin()+half*4,f.corners.begin()+half*4+4);
                polylines(overlay,points,true,half?Scalar(0,220,255):Scalar(255,220,0),2,LINE_AA);
            }
            putText(overlay,std::to_string(i),f.feature_center,FONT_HERSHEY_SIMPLEX,.4,Scalar(0,0,255),1,LINE_AA);
        }
        stageImages.push_back({"04_pairs",overlay});
    }
	if (features.size() < this->featureSize) {
		return;
	}


	detector.markerOrganization(features, markers);
    candidateMarkers = markers;
    stats.groups = markers.size();
    if(debugCapture) {
        Mat overlay; cvtColor(img,overlay,COLOR_GRAY2BGR);
        for(size_t i=0;i<markers.size();++i) {
            const auto color=Scalar(70+(i*73)%180,70+(i*47)%180,70+(i*113)%180);
            for(size_t k=0;k<markers[i].cornerLists.size();++k) {
                const auto& p=markers[i].cornerLists[k];
                for(int half=0;half<2;++half) {
                    vector<Point> points(p.begin()+half*4,p.begin()+half*4+4);
                    polylines(overlay,points,true,color,2,LINE_AA);
                }
                putText(overlay,std::to_string(markers[i].feature_ID[k]),markers[i].feature_center[k],FONT_HERSHEY_SIMPLEX,.4,Scalar(0,0,255),1,LINE_AA);
            }
        }
        stageImages.push_back({"05_groups",overlay});
    }
    for (const auto& candidate : markers)
        for (int code : candidate.feature_ID) stats.validColumns += code >= 0;

	detector.markerDecoder(markers, markers, this->state, this->featureSize);
	markers_info = markers;

}

void CylinderTag::loadModel(const string& path, vector<ModelInfo>& reconstruct_model){
	ifstream input_file(path);
	if (!input_file.is_open())
	{
		throw __FUNCTION__ + string(", ") + "could not open the model file\n";
	}

	int model_num, model_size, corner_id, model_ID;
	input_file >> model_num >> model_size;
	reconstruct_model.resize(model_num);
	Point3f world_point;
	for (int i = 0; i < model_num; i++) {
		input_file >> model_ID;
		reconstruct_model[i].MarkerID = model_ID;
		input_file >> reconstruct_model[i].base.x;
		input_file >> reconstruct_model[i].base.y;
		input_file >> reconstruct_model[i].base.z;
		input_file >> reconstruct_model[i].axis.x;
		input_file >> reconstruct_model[i].axis.y;
		input_file >> reconstruct_model[i].axis.z;
		reconstruct_model[i].corners.resize(model_size * 8);
		for (int j = 0; j < 8 * model_size; j++) {
			input_file >> corner_id;
			input_file >> reconstruct_model[i].corners[corner_id].x;
			input_file >> reconstruct_model[i].corners[corner_id].y;
			input_file >> reconstruct_model[i].corners[corner_id].z;
		}		
	}
	
}

void CylinderTag::loadCamera(const string& path, CamInfo& camera){
	FileStorage fs(path, FileStorage::READ);
	fs["cameraMatrix"] >> camera.Intrinsic;
	fs["distCoeffs"] >> camera.distCoeffs;
}

void CylinderTag::estimatePose(const Mat& img, vector<MarkerInfo> markers, vector<ModelInfo> reconstruct_model, CamInfo camera, vector<PoseInfo>& pose, bool useDensePoseRefine){
	pose.resize(markers.size());
	for (int i = 0; i < markers.size(); i++) {
		estimator.PnPSolver(markers[i], reconstruct_model, camera, pose[i]);
		if (useDensePoseRefine)
			estimator.DenseSolver(img, reconstruct_model, pose[i]);
	}

	pose.erase(std::remove_if(pose.begin(), pose.end(), [](PoseInfo const& pose) {
		return pose.markerID == -1;
	}), pose.end());
}

void CylinderTag::drawAxis(const Mat& img, vector<MarkerInfo> markers, vector<ModelInfo> reconstruct_model, vector<PoseInfo>& pose, CamInfo camera, int axisLength = 5){
	// Display
	Mat imgMark(img.rows, img.cols, CV_32FC3);
	cvtColor(img, imgMark, COLOR_GRAY2RGB);

	vector<Point2f> imagePoints, image_points;
	vector<Point3f> model_points;
	for (int i = 0; i < pose.size(); i++) {
		model_points.clear();
		image_points.clear();
		int ID = pose[i].markerID;
		for (int j = 0; j < markers[i].cornerLists.size(); j++) {
			for (int k = 0; k < 8; k++) {
				model_points.push_back(reconstruct_model[ID].corners[markers[i].featurePos[j] * 8 + k]);
				image_points.push_back(markers[i].cornerLists[j][k]);
			}
		}
		model_points.push_back(reconstruct_model[ID].base);
		model_points.push_back(reconstruct_model[ID].base + reconstruct_model[ID].axis * axisLength);
		model_points.push_back(reconstruct_model[ID].base + Point3f(0.0372, 0.0372, 0.9986) * axisLength);
		model_points.push_back(reconstruct_model[ID].base + Point3f(0.9980, -0.0520, -0.0353) * axisLength);
					
		imagePoints.clear();
		projectPoints(model_points, pose[i].rvec, pose[i].tvec, camera.Intrinsic, camera.distCoeffs, imagePoints);
		for (int i = 0; i < imagePoints.size() - 5; i++) {
			circle(imgMark, imagePoints[i], 5, Scalar(255, 234, 32), -1);
		}

		arrowedLine(imgMark, imagePoints[imagePoints.size() - 4], imagePoints[imagePoints.size() - 3], Scalar(255, 0, 0), 10, LINE_AA, 0, 0.2);
		arrowedLine(imgMark, imagePoints[imagePoints.size() - 4], imagePoints[imagePoints.size() - 2], Scalar(0, 255, 0), 10, LINE_AA, 0, 0.2);
		arrowedLine(imgMark, imagePoints[imagePoints.size() - 4], imagePoints[imagePoints.size() - 1], Scalar(0, 0, 255), 10, LINE_AA, 0, 0.2);
		circle(imgMark, imagePoints[imagePoints.size() - 4], 8, Scalar(247, 235, 235), -1);
	}
	imshow("Plot", imgMark);
	waitKey(1);
}
