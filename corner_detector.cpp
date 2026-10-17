#include "header/corner_detector.h"

using namespace std;
using namespace cv;

template <typename T>
std::vector<size_t> sort_indexes_greater(const std::vector<T> &v) {
    // 初始化索引向量
    std::vector<size_t> idx(v.size());
    //使用iota对向量赋0~n的连续值
    std::iota(idx.begin(), idx.end(), 0);
    // 通过比较v的值对索引idx进行排序
    std::stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });
    return idx;
}

template <typename T>
std::vector<size_t> sort_indexes_lesser(const std::vector<T>& v) {
    // 初始化索引向量
    std::vector<size_t> idx(v.size());
    //使用iota对向量赋0~n的连续值
    std::iota(idx.begin(), idx.end(), 0);
    // 通过比较v的值对索引idx进行排序
    std::stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
    return idx;
}

void corner_detector::adaptiveThreshold(const Mat& src, Mat& dst, int thresholdWindow){
    int col_number = src.cols / thresholdWindow + (src.cols % thresholdWindow != 0 ? 1 : 0);
    int row_number = src.rows / thresholdWindow + (src.rows % thresholdWindow != 0 ? 1 : 0);
    Mat extreme_min(row_number, col_number, CV_32FC1);
    Mat extreme_max(row_number, col_number, CV_32FC1);
    Mat extreme_min_final(row_number, col_number, CV_32FC1);
    Mat extreme_max_final(row_number, col_number, CV_32FC1);

    RAC = 0;
    row_count = 0;
    col_count = 0;
    row_count_final = 1;
    col_count_final = 1;

    for (int i = 0; i < src.rows; i += thresholdWindow){
        for (int j = 0; j < src.cols; j += thresholdWindow){
            Rect rect(j, i, min(thresholdWindow, src.cols - j), min(thresholdWindow, src.rows - i));
            img_part = src(rect);
            minMaxLoc(img_part, &minVal, &maxVal, NULL, NULL);
            extreme_min.at<float>(row_count, col_count) = (float)minVal;
            extreme_max.at<float>(row_count, col_count) = (float)maxVal;
            col_count++;
        }
        col_count = 0;
        row_count++;
    }
    // Initialize border tiles too. The former interior-only loop read
    // uninitialized thresholds at all four image boundaries.
    erode(extreme_min, extreme_min_final, Mat::ones(3,3,CV_8UC1));
    dilate(extreme_max, extreme_max_final, Mat::ones(3,3,CV_8UC1));

    for (int i = 0; i < src.rows; i++){
        for (int j = 0; j < src.cols; j++){
            const float lo=extreme_min_final.at<float>(i/thresholdWindow,j/thresholdWindow);
            const float hi=extreme_max_final.at<float>(i/thresholdWindow,j/thresholdWindow);
            if (hi-lo >= .05f && src.at<float>(i,j) < (hi+lo)/2){
                dst.at<uchar>(i, j) = 255;
            }
            else{
                dst.at<uchar>(i, j) = 0;
            }
        }
    }
}

void corner_detector::connectedComponentLabeling(const Mat& src, vector<vector<Point>>& quadArea, int method){
    nccomp_area = connectedComponentsWithStats(src, img_labeled, stats, centroids, 8, 4, CCL_BBDT);
    quadArea.resize(nccomp_area);
    illegal.resize(nccomp_area);
    fill(illegal.begin(), illegal.end(), 0);

    for (int i = 0; i < nccomp_area; i++){
        if (stats.at<int>(i, cv::CC_STAT_AREA) < 30 || stats.at<int>(i, cv::CC_STAT_AREA) > round(0.01 * src.cols * src.rows)){
            illegal[i] = true;
        }
    }
    for (int i = 0; i < img_labeled.rows; i++){
        for (int j = 0; j < img_labeled.cols; j++){
            if (!illegal[img_labeled.at<int>(i, j)])
                quadArea[img_labeled.at<int>(i, j)].push_back(Point(j, i));
        }
    }
    int count = 0;
    for (auto iter = quadArea.begin(); iter != quadArea.end(); ){
        if (illegal[count++]){
            iter = quadArea.erase(iter);
        }
        else{
            iter++;
        }
    }
}

inline bool comp_x(Point& a, Point& b){
    return a.x > b.x;
}

inline bool comp_y(Point& a, Point& b){
    return a.y > b.y;
}

inline bool cmp_dis(const corners_pre& a, corners_pre& b){
    return a.dis_to_centroid < b.dis_to_centroid;
}

inline bool cmp_ang(const corners_pre& a, corners_pre& b){
    return a.angle_to_centroid < b.angle_to_centroid;
}

vector<int> corner_detector::expand_line(vector<Point> edge_point, int init, int end){
    span_temp.clear();
    find_edge_left = false;
    find_edge_right = false;
    left = init - 1;
    right = end + 1;
    vector<Point>::const_iterator First = edge_point.begin() + init;
    vector<Point>::const_iterator Middle = edge_point.begin() + end + 1;
    vector<Point> Slide;
    Slide.assign(First, Middle);
    vector<float> line;
    fitLine(Slide, line, DIST_L2, 0, 0.01, 0.01);

    for (int i = init; i <= end; i++){
        span_temp.push_back(i);
    }
    while ((!find_edge_left || !find_edge_right) && (left != right)){
        if (!find_edge_left){
            if (left == -1) left = edge_point.size() - 1;
            dist_expand = abs(edge_point[left].x * line[1] - edge_point[left].y * line[0] + line[0] * line[3] - line[1] * line[2]);
            if (dist_expand > threshold_expand){
                find_edge_left = true;
                continue;
            }
            Slide.push_back(edge_point[left]);
            span_temp.push_back(left--);
            fitLine(Slide, line, DIST_L2, 0, 0.01, 0.01);
            if (Slide.size() == edge_point.size()) break;
        }
        if (!find_edge_right){
            if (right == edge_point.size()) right = 0;
            dist_expand = abs(edge_point[right].x * line[1] - edge_point[right].y * line[0] + line[0] * line[3] - line[1] * line[2]);
            if (dist_expand > threshold_expand){
                find_edge_right = true;
                continue;
            }
            Slide.push_back(edge_point[right]);
            span_temp.push_back(right++);
            fitLine(Slide, line, DIST_L2, 0, 0.01, 0.01);
            if (Slide.size() == edge_point.size()) break;
        }
    }
    sort(span_temp.begin(), span_temp.end(), greater<int>());
    return span_temp;
}

// AprilTag-style edge refinement: aggregate intensity transitions along each
// side, fit a line to many samples, and intersect adjacent fitted lines.
void corner_detector::refineQuads(const Mat& gray, vector<vector<Point2f>>& quads, int searchRadius) {
    auto sample=[&](Point2f p) -> float {
        if(p.x<0||p.y<0||p.x>=gray.cols-1||p.y>=gray.rows-1) return NAN;
        const int x=int(p.x),y=int(p.y);
        const float a=p.x-x,b=p.y-y;
        return (1-b)*((1-a)*gray.at<float>(y,x)+a*gray.at<float>(y,x+1))+
                b*((1-a)*gray.at<float>(y+1,x)+a*gray.at<float>(y+1,x+1));
    };
    for(auto& quad : quads) {
        array<Vec4f,4> lines;
        bool valid=true;
        float shortest=1e9;
        for(int i=0;i<4;++i) shortest=std::min(shortest,float(norm(quad[i]-quad[(i+1)%4])));
        const float range=std::min(float(searchRadius),std::max(1.f,shortest*.25f));
        for(int side=0;side<4;++side) {
            const Point2f a=quad[side],b=quad[(side+1)%4],delta=b-a;
            const float length=norm(delta);
            const Point2f normal(delta.y/length,-delta.x/length);
            vector<Point2f> edgePoints;
            const int samples=std::clamp(int(length),8,80);
            for(int k=0;k<samples;++k) {
                const float t=.15f+.7f*(k+.5f)/samples;
                const Point2f center=a+t*delta;
                float best=-1,offset=0;
                for(float shift=-range;shift<=range;shift+=.25f) {
                    const float gradient=sample(center+(shift+.65f)*normal)-sample(center+(shift-.65f)*normal);
                    if(std::isfinite(gradient)&&gradient>best) {best=gradient;offset=shift;}
                }
                if(best<.06f) continue;
                float weighted=0,weight=0;
                for(float shift=offset-.5f;shift<=offset+.5f;shift+=.125f) {
                    const float gradient=sample(center+(shift+.65f)*normal)-sample(center+(shift-.65f)*normal);
                    if(std::isfinite(gradient)&&gradient>0) {weighted+=shift*gradient;weight+=gradient;}
                }
                if(weight>0) edgePoints.push_back(center+normal*(weighted/weight));
            }
            if(edgePoints.size()<4) {valid=false;break;}
            fitLine(edgePoints,lines[side],DIST_WELSCH,0,.01,.01);
        }
        if(!valid) continue;
        auto next=quad;
        for(int k=0;k<4;++k) {
            const auto& a=lines[(k+3)%4];const auto& b=lines[k];
            const double determinant=a[0]*b[1]-a[1]*b[0];
            if(abs(determinant)<.05) {valid=false;break;}
            const double t=((b[2]-a[2])*b[1]-(b[3]-a[3])*b[0])/determinant;
            next[k]={float(a[2]+t*a[0]),float(a[3]+t*a[1])};
            if(!std::isfinite(next[k].x)||!std::isfinite(next[k].y)||norm(next[k]-quad[k])>range*2) {valid=false;break;}
        }
        if(valid&&isContourConvex(next)) quad=next;
    }
}

// ArUco-style candidate generation: closed contours, polygon approximation,
// convexity and scale checks. Keep the measured boundary, not a bounding box.
void corner_detector::contourQuads(const Mat& binary, vector<vector<Point2f>>& quads) {
    vector<vector<Point>> contours;
    findContours(binary,contours,RETR_EXTERNAL,CHAIN_APPROX_NONE);
    quads.clear();
    for(const auto& contour : contours) {
        const double area=abs(contourArea(contour));
        if(area<20||area>binary.total()*.04) continue;
        const double perimeter=arcLength(contour,true);
        vector<Point> poly;
        for(double accuracy : {.015,.025,.04}) {
            approxPolyDP(contour,poly,accuracy*perimeter,true);
            if(poly.size()==4) break;
        }
        if(poly.size()!=4||!isContourConvex(poly)) continue;
        bool valid=true;
        for(int k=0;k<4;++k) {
            valid &= norm(poly[k]-poly[(k+1)%4])>=3;
            valid &= poly[k].x>1&&poly[k].y>1&&poly[k].x<binary.cols-2&&poly[k].y<binary.rows-2;
        }
        if(!valid) continue;
        vector<Point2f> points(poly.begin(),poly.end());
        // Pairing expects the same winding as the legacy edge extractor.
        if(contourArea(points,true)<0) reverse(points.begin(),points.end());
        quads.push_back(points);
    }
}

void corner_detector::edgeExtraction(const Mat& img, vector<vector<Point>>& quadArea, vector<vector<Point2f>>& corners_init){
    Mat A(2, 2, CV_32FC1);
    Mat B(2, 1, CV_32FC1);
    Mat sol(2, 1, CV_32FC1);

    for (int i = 0; i < quadArea.size(); i++){
        edge_point.clear();
        corners_p.clear();
        for (int j = 0; j < 4; j++){
            edge_point_cluster[j].clear();
        }

        // Ray-casting Algorithm
        sort(quadArea[i].begin(), quadArea[i].end(), comp_x);
        x_max = quadArea[i][0].x;
        x_min = quadArea[i][quadArea[i].size() - 1].x;
        sort(quadArea[i].begin(), quadArea[i].end(), comp_y);
        y_max = quadArea[i][0].y;
        y_min = quadArea[i][quadArea[i].size() - 1].y;

        Mat mask = Mat::zeros(y_max - y_min + 1, x_max - x_min + 1, CV_8UC1);
        Mat visited = Mat::zeros(y_max - y_min + 1, x_max - x_min + 1, CV_8UC1);
        for (int j = 0; j < quadArea[i].size(); j++){
            mask.at<uchar>(quadArea[i][j].y - y_min, quadArea[i][j].x - x_min) = 1;
        }

        //Upon search
        for (int j = 0; j < mask.cols; j++)
            for (int k = 0; k < mask.rows; k++){
                if (visited.at<uchar>(k, j)) break;
                if (mask.at<uchar>(k, j)){
                    visited.at<uchar>(k, j) = 1;
                    break;
                }
            }
        //Down search
        for (int j = 0; j < mask.cols; j++)
            for (int k = mask.rows - 1; k >= 0; k--){
                if (visited.at<uchar>(k, j)) break;
                if (mask.at<uchar>(k, j)){
                    visited.at<uchar>(k, j) = 1;
                    break;
                }
            }
        //Left search
        for (int k = 0; k < mask.rows; k++)
            for (int j = 0; j < mask.cols; j++){
                if (visited.at<uchar>(k, j)) break;
                if (mask.at<uchar>(k, j)){
                    visited.at<uchar>(k, j) = 1;
                    break;
                }
            }
        //Right search
        for (int k = 0; k < mask.rows; k++)
            for (int j = mask.cols - 1; j >= 0; j--){
                if (visited.at<uchar>(k, j)) break;
                if (mask.at<uchar>(k, j)){
                    visited.at<uchar>(k, j) = 1;
                    break;
                }
            }

        //Find start point
        for (int j = 0; j < visited.cols; j++)
            for (int k = 0; k < visited.rows; k++){
                if (visited.at<uchar>(k, j)){
                    starter = Point(j, k);
                    edge_point.push_back(Point(j + x_min, k + y_min));
                    visited.at<uchar>(k, j) = 0;
                    j = visited.cols;
                    break;
                }
            }

        edge_number = sum(visited);
        get_orientedEdgePoints(visited, starter, edge_number[0]);

        //Extract the center of the boundaries
        sum_x = 0, sum_y = 0;
        for (int j = 0; j < edge_point.size(); j++){
            sum_x += edge_point[j].x;
            sum_y += edge_point[j].y;
        }
        area_center.x = 1.0 * sum_x / edge_point.size();
        area_center.y = 1.0 * sum_y / edge_point.size();

        //Calculate the distance from boundary to center
        dist2center.clear();
        for (int j = 0; j < edge_point.size(); j++){
            dist2center.push_back(distance_2points(edge_point[j], area_center));
        }
        auto b = sort_indexes_lesser<float>(dist2center);
        if (b[0] > 0){
            vector<Point>::const_iterator First = edge_point.begin() + b[0];
            vector<Point>::const_iterator Second = edge_point.end();
            vector<Point>::const_iterator Begin = edge_point.begin();
            vector<Point>::const_iterator Middle = edge_point.begin() + b[0];
            vector<Point> Slide1, Slide2;
            Slide1.assign(First, Second);
            Slide2.assign(Begin, Middle);
            edge_point.clear();
            edge_point.insert(edge_point.end(),Slide1.begin(),Slide1.end());
            edge_point.insert(edge_point.end(),Slide2.begin(),Slide2.end());
        }

        // Extended Ramer-Douglas-Peucker Algorithm
        cnt_boundary = 0;
        init = 0;
        bool isFailed = false;

        while (!(edge_point.empty()) && !isFailed && cnt_boundary < 4){
            //Judge triplet
            if (edge_point.size() > 2){
                cost = norm(edge_point[init] + edge_point[(init + 2) % edge_point.size()] - 2 * edge_point[(init + 1) % edge_point.size()]);
                while ((cost > 1.05) && (init < edge_point.size() - 3)){
                    init++;
                    cost = norm(edge_point[init] + edge_point[(init + 2) % edge_point.size()] - 2 * edge_point[(init + 1) % edge_point.size()]);
                }
            }
            else {
                isFailed = true;
                break;
            }

            //Obtain middle point
            int end = init + edge_point.size() / 2;
            if (end > edge_point.size() - 1){
                end = edge_point.size() - 1;
            }

            //Main algorithm
            while (1)
            {
                if (end <= init + 1){
                    isFailed = true;
                    break;
                }
                if (edge_point[init].x == edge_point[end].x){
                    normal_line[0] = 100;
                    normal_line[1] = -1;
                }else{
                    normal_line[0] = 1.0 * (edge_point[end].y - edge_point[init].y) / (edge_point[end].x - edge_point[init].x);
                    normal_line[1] = -1;
                }
                d_line = -(normal_line[0] * edge_point[init].x + normal_line[1] * edge_point[init].y);

                dist2line.clear();
                for (int iter = init + 1; iter < end; iter++){
                    dist2line.push_back(abs(normal_line[0] * edge_point[iter].x + normal_line[1] * edge_point[iter].y + d_line) / sqrt(normal_line[0] * normal_line[0] + 1));
                }
                b = sort_indexes_greater<float>(dist2line);
                if ((dist2line[b[0]] > threshold_line) && (b.size() > 1)){
                    int find_max_end = 1;
                    while ((find_max_end < b.size()) && (dist2line[b[0]] == dist2line[b[find_max_end]]))
                    {
                        find_max_end++;
                    }
                    end = b[find_max_end - 1];
                }else{
                    span = expand_line(edge_point, init, end);
                    for (int iter = 0; iter < span.size(); iter++){
                        edge_point_cluster[cnt_boundary].push_back(edge_point[span[iter]]);
                    }

                    //endpoint keeping judgment
                    cost = norm(edge_point[span[0]] + edge_point[(span[0] + 2) % edge_point.size()] - 2 * edge_point[(span[0] + 1) % edge_point.size()]);
                    if (cost < 1.05)
                        span.erase(span.begin());

                    for (int iter = 0; iter < span.size(); iter++){
                        edge_point.erase(edge_point.begin() + span[iter]);
                    }
                    cnt_boundary++;
                    init = span[span.size() - 1] >= edge_point.size() ? 0 : span[span.size() - 1];
                    break;
                }
            }
        }

        flag_line_number = false;
        for (int j = 0; j < 4; j++){
            // No enough points to fit a line
            if (edge_point_cluster[j].size() < 2){
                flag_line_number = true;
                break;
            }
            fitLine(edge_point_cluster[j], line_func[j], DIST_WELSCH, 0, 0.01, 0.01);
        }

        if (flag_line_number) continue;
        for (int j = 0; j < 3; j++){
            for (int k = j + 1; k < 4; k++){
                A.at<float>(0, 0) = line_func[j][1];
                A.at<float>(0, 1) = -line_func[j][0];
                A.at<float>(1, 0) = line_func[k][1];
                A.at<float>(1, 1) = -line_func[k][0];
                B.at<float>(0, 0) = line_func[j][1] * line_func[j][2] - line_func[j][0] * line_func[j][3];
                B.at<float>(1, 0) = line_func[k][1] * line_func[k][2] - line_func[k][0] * line_func[k][3];
                if (determinant(A) != 0){
                    solve(A, B, sol);
                    c.intersect.x = sol.at<float>(0, 0);
                    c.intersect.y = sol.at<float>(1, 0);
                    c.dis_to_centroid = sqrt((c.intersect.x - area_center.x) * (c.intersect.x - area_center.x) + (c.intersect.y - area_center.y) * (c.intersect.y - area_center.y));
                    c.angle_to_centroid = atan2(c.intersect.y - area_center.y, c.intersect.x - area_center.x) * 180 / CV_PI;
                    if (c.dis_to_centroid < img.cols && c.dis_to_centroid < img.rows)
                        corners_p.push_back(c);
                }
            }
        }

        sort(corners_p.begin(), corners_p.end(), cmp_ang);
        memset(re, 0, sizeof(re));
        memset(vis, 0, sizeof(vis));
        rac_min = threshold_RAC;
        corners_p = get_permutation(0, 0, corners_p, quadArea[i].size());
        if (corners_p.size() != 4) continue;

        flag_illegal_corner = false;
        for (int j = 0; j < 4; j++){
            if (corners_p[j].intersect.x < 0 || corners_p[j].intersect.y < 0 || corners_p[j].intersect.x > img.cols || corners_p[j].intersect.y > img.rows){
                flag_illegal_corner = true;
                break;
            }
        }
        if (flag_illegal_corner) continue;

        corners_pass.clear();
        for (int j = 0; j < 4; j++){
            corners_pass.push_back(corners_p[j].intersect);
        }

        corners_init.push_back(corners_pass);
    }
}

void corner_detector::get_orientedEdgePoints(Mat& visited, Point starter, int count)
{
    for (int j = 0; j < 8; j++) {
        if (starter.y + y_bias[j] >= 0 && starter.y + y_bias[j] < visited.rows && starter.x + x_bias[j] >= 0 && starter.x + x_bias[j] < visited.cols)
            if (visited.at<uchar>(starter.y + y_bias[j], starter.x + x_bias[j])) {
                edge_point.push_back(Point(starter.x + x_bias[j] + x_min, starter.y + y_bias[j] + y_min));
                visited.at<uchar>(starter.y + y_bias[j], starter.x + x_bias[j]) = 0;
                starter = Point(starter.x + x_bias[j], starter.y + y_bias[j]);
                get_orientedEdgePoints(visited, starter, count - 1);
            }
    }
}

vector<corners_pre> corner_detector::get_permutation(int step, int start, vector<corners_pre>& corners, int area)
{
    if (step == 4)
    {
        corners_per.clear();
        for (int i = 0; i < 4; i++) {
            corners_per.push_back(corners[re[i]]);
        }
        float s1 = corners_per[0].intersect.x * corners_per[1].intersect.y + corners_per[1].intersect.x * corners_per[2].intersect.y + corners_per[2].intersect.x * corners_per[0].intersect.y - corners_per[0].intersect.x * corners_per[2].intersect.y - corners_per[1].intersect.x * corners_per[0].intersect.y - corners_per[2].intersect.x * corners_per[1].intersect.y;
        float s2 = corners_per[1].intersect.x * corners_per[2].intersect.y + corners_per[2].intersect.x * corners_per[3].intersect.y + corners_per[3].intersect.x * corners_per[1].intersect.y - corners_per[1].intersect.x * corners_per[3].intersect.y - corners_per[2].intersect.x * corners_per[1].intersect.y - corners_per[3].intersect.x * corners_per[2].intersect.y;
        float s3 = corners_per[2].intersect.x * corners_per[3].intersect.y + corners_per[3].intersect.x * corners_per[0].intersect.y + corners_per[0].intersect.x * corners_per[2].intersect.y - corners_per[2].intersect.x * corners_per[0].intersect.y - corners_per[3].intersect.x * corners_per[2].intersect.y - corners_per[0].intersect.x * corners_per[3].intersect.y;
        float s4 = corners_per[0].intersect.x * corners_per[1].intersect.y + corners_per[1].intersect.x * corners_per[3].intersect.y + corners_per[3].intersect.x * corners_per[0].intersect.y - corners_per[0].intersect.x * corners_per[3].intersect.y - corners_per[1].intersect.x * corners_per[0].intersect.y - corners_per[3].intersect.x * corners_per[1].intersect.y;
        if (abs(s1) < 1 || abs(s2) < 1 || abs(s3) < 1 || abs(s4) < 1) {
            corners_per.clear();
            return corners_per;
        }
        rac_now = quadJudgment(corners_per, area);
        if (rac_now < rac_min) {
            rac_min = rac_now;
            corners_final = corners_per;
        }
    }
    for (int i = start; i < corners.size(); i++)
    {
        if (vis[i]) continue;
        vis[i] = true;
        re[step] = i;
        get_permutation(step + 1, i + 1, corners, area);
        vis[i] = false;
    }
    if (rac_min >= threshold_RAC)  corners_final.clear();
    return corners_final;
}

float corner_detector::quadJudgment(vector<corners_pre>& corners, int areaPixelNumber){
    quad_area = 0;
    for (int i = 0; i < 3; i++){
        quad_area += corners[i].intersect.x * corners[i + 1].intersect.y - corners[i].intersect.y * corners[i + 1].intersect.x;
    }
    quad_area += corners[3].intersect.x * corners[0].intersect.y - corners[3].intersect.y * corners[0].intersect.x;
    quad_area /= 2;
    RAC = abs(abs(quad_area) - areaPixelNumber) / areaPixelNumber;
    return RAC;
}

void corner_detector::featureRecovery(vector<vector<Point2f>>& corners_refined, vector<featureInfo>& features){
    isVisited.assign(corners_refined.size(), false);
    struct PairCandidate {int first,second;float score;featureInfo feature;};
    vector<PairCandidate> candidates;

    corner_dist.resize(corners_refined.size());
    corner_centers.clear();
    corner_angles_1.clear();
    corner_angles_2.clear();

    for (int i = 0; i < corners_refined.size(); i++){
        Point2f center_ = Point2f((corners_refined[i][0].x + corners_refined[i][1].x + corners_refined[i][2].x + corners_refined[i][3].x) / 4, (corners_refined[i][0].y + corners_refined[i][1].y + corners_refined[i][2].y + corners_refined[i][3].y) / 4);
        corner_centers.push_back(center_);
        for (int j = 0; j < 4; j++){
            corner_dist[i][j] = sqrt((corners_refined[i][j].x - corners_refined[i][(j + 1) % 4].x) * (corners_refined[i][j].x - corners_refined[i][(j + 1) % 4].x) + (corners_refined[i][j].y - corners_refined[i][(j + 1) % 4].y) * (corners_refined[i][j].y - corners_refined[i][(j + 1) % 4].y));
        }
        corner_angles_1.push_back((atan2(corners_refined[i][0].y - corners_refined[i][1].y, corners_refined[i][0].x - corners_refined[i][1].x) * 180 / CV_PI + atan2(corners_refined[i][3].y - corners_refined[i][2].y, corners_refined[i][3].x - corners_refined[i][2].x) * 180 / CV_PI) / 2);
        corner_angles_2.push_back((atan2(corners_refined[i][1].y - corners_refined[i][2].y, corners_refined[i][1].x - corners_refined[i][2].x) * 180 / CV_PI + atan2(corners_refined[i][0].y - corners_refined[i][3].y, corners_refined[i][0].x - corners_refined[i][3].x) * 180 / CV_PI) / 2);
    }
    for (int i = 0; i < corners_refined.size() - 1; i++){
        if (isVisited[i]) continue;
        for (int j = i + 1; j < corners_refined.size(); j++) {
            if (!isVisited[j]){
                tag1 = false;
                tag2 = false;
                feature_angle = atan2(corner_centers[i].y - corner_centers[j].y, corner_centers[i].x - corner_centers[j].x) * 180 / CV_PI;

                if (abs(feature_angle - corner_angles_1[i]) < threshold_angle || abs(abs(feature_angle - corner_angles_1[i]) - 180) < threshold_angle || abs(abs(feature_angle - corner_angles_1[i]) - 360) < threshold_angle){
                    tag1 = true;
                    dist1_long = (corner_dist[i][0] + corner_dist[i][2]) / 2;
                    dist1_short = min(corner_dist[i][1], corner_dist[i][3]);
                    if (corner_dist[i][1] < corner_dist[i][3]) {
                        edge_angle1 = atan2(corners_refined[i][0].y - corners_refined[i][3].y, corners_refined[i][0].x - corners_refined[i][3].x) * 180 / CV_PI;
                        feature_point1 = (corners_refined[i][1] + corners_refined[i][2]) / 2;
                    }
                    else {
                        edge_angle1 = atan2(corners_refined[i][1].y - corners_refined[i][2].y, corners_refined[i][1].x - corners_refined[i][2].x) * 180 / CV_PI;
                        feature_point1 = (corners_refined[i][0] + corners_refined[i][3]) / 2;
                    }
                }
                if (abs(feature_angle - corner_angles_2[i]) < threshold_angle || abs(abs(feature_angle - corner_angles_2[i]) - 180) < threshold_angle || abs(abs(feature_angle - corner_angles_2[i]) - 360) < threshold_angle){
                    tag1 = true;
                    dist1_short = min(corner_dist[i][0], corner_dist[i][2]);
                    dist1_long = (corner_dist[i][1] + corner_dist[i][3]) / 2;
                    if (corner_dist[i][0] > corner_dist[i][2]) {
                        edge_angle1 = atan2(corners_refined[i][0].y - corners_refined[i][1].y, corners_refined[i][0].x - corners_refined[i][1].x) * 180 / CV_PI;
                        feature_point1 = (corners_refined[i][2] + corners_refined[i][3]) / 2;
                    }
                    else {
                        edge_angle1 = atan2(corners_refined[i][2].y - corners_refined[i][3].y, corners_refined[i][2].x - corners_refined[i][3].x) * 180 / CV_PI;
                        feature_point1 = (corners_refined[i][0] + corners_refined[i][1]) / 2;
                    }
                }
                if (abs(feature_angle - corner_angles_1[j]) < threshold_angle || abs(abs(feature_angle - corner_angles_1[j]) - 180) < threshold_angle || abs(abs(feature_angle - corner_angles_1[j]) - 360) < threshold_angle){
                    tag2 = true;
                    dist2_long = (corner_dist[j][0] + corner_dist[j][2]) / 2;
                    dist2_short = min(corner_dist[j][1], corner_dist[j][3]);
                    if (corner_dist[j][1] < corner_dist[j][3]) {
                        edge_angle2 = atan2(corners_refined[j][0].y - corners_refined[j][3].y, corners_refined[j][0].x - corners_refined[j][3].x) * 180 / CV_PI;
                        feature_point2 = (corners_refined[j][1] + corners_refined[j][2]) / 2;
                    }
                    else {
                        edge_angle2 = atan2(corners_refined[j][1].y - corners_refined[j][2].y, corners_refined[j][1].x - corners_refined[j][2].x) * 180 / CV_PI;
                        feature_point2 = (corners_refined[j][0] + corners_refined[j][3]) / 2;
                    }
                }
                if (abs(feature_angle - corner_angles_2[j]) < threshold_angle || abs(abs(feature_angle - corner_angles_2[j]) - 180) < threshold_angle || abs(abs(feature_angle - corner_angles_2[j]) - 360) < threshold_angle){
                    tag2 = true;
                    dist2_short = min(corner_dist[j][0], corner_dist[j][2]);
                    dist2_long = (corner_dist[j][1] + corner_dist[j][3]) / 2;
                    if (corner_dist[j][0] > corner_dist[j][2]) {
                        edge_angle2 = atan2(corners_refined[j][0].y - corners_refined[j][1].y, corners_refined[j][0].x - corners_refined[j][1].x) * 180 / CV_PI;
                        feature_point2 = (corners_refined[j][2] + corners_refined[j][3]) / 2;
                    }
                    else {
                        edge_angle2 = atan2(corners_refined[j][2].y - corners_refined[j][3].y, corners_refined[j][2].x - corners_refined[j][3].x) * 180 / CV_PI;
                        feature_point2 = (corners_refined[j][0] + corners_refined[j][1]) / 2;
                    }
                }
                feature_length = distance_2points(corner_centers[i], corner_centers[j]);
                if ((tag1 && tag2) && (dist1_long > dist1_short || dist2_long > dist2_short)
                    && (abs(edge_angle1 - edge_angle2) < threshold_angle * 10 || abs(abs(edge_angle1 - edge_angle2) - 180) < threshold_angle * 10 || abs(abs(edge_angle1 - edge_angle2) - 360) < threshold_angle * 10)
                    && (abs(dist1_short - dist2_short) < min(dist1_short, dist2_short) * 0.33)
                    && ((dist1_long + dist2_long) > (dist1_short + dist2_short))
                    && ((dist1_long + dist2_long) < 15 * (dist1_short + dist2_short))
                    && (feature_length - (dist1_long + dist2_long) / 2 < 0.3 * (feature_length + (dist1_long + dist2_long) / 2)))
                {
                    const auto feature=featureOrganization(corners_refined[i], corners_refined[j], corner_centers[i], corner_centers[j], feature_angle);
                    const auto& p=feature.corners;
                    const float width=(norm(p[0]-p[1])+norm(p[4]-p[5]))/2;
                    float rms=0;
                    for(const vector<int>& indices : {vector<int>{0,3,6,5},vector<int>{1,2,7,4}}) {
                        vector<Point2f> points;
                        for(int k:indices) points.push_back(p[k]);
                        Vec4f line; fitLine(points,line,DIST_L2,0,.01,.01);
                        for(const auto& q:points) {
                            const float d=(q.x-line[2])*line[1]-(q.y-line[3])*line[0];
                            rms+=d*d;
                        }
                    }
                    rms=std::sqrt(rms/8);
                    // The corresponding long edges lie on cylinder
                    // generatrices. Neighboring bars fail this collinearity
                    // check even when their centroids happen to align.
                    if(rms<=std::max(.8f,.1f*width))
                        candidates.push_back({i,j,rms/std::max(1.f,width),feature});
                }
            }
        }
    }

    std::stable_sort(candidates.begin(),candidates.end(),[](const auto& a,const auto& b){return a.score<b.score;});
    for(const auto& candidate:candidates) {
        if(isVisited[candidate.first]||isVisited[candidate.second]) continue;
        isVisited[candidate.first]=isVisited[candidate.second]=true;
        features.push_back(candidate.feature);
    }

}

void corner_detector::cornerObtain(const Mat& src, vector<featureInfo>& features){
    TermCriteria termcrit(TermCriteria::COUNT + TermCriteria::EPS, 20, 0.1);
    for (int i = 0; i < features.size(); i++) {
        for (int j = 0; j < features[i].corners.size(); j++) {
            features[i].corners[j] = (features[i].corners[j] - Point2f(0.5, 0.5)) * 2 + Point2f(0.5, 0.5);
        }
        features[i].feature_center = (features[i].corners[0] + features[i].corners[1] + features[i].corners[4] + features[i].corners[5]) / 4;
    }
}

featureInfo corner_detector::featureOrganization(vector<Point2f> quad1, vector<Point2f> quad2, Point2f quad1_center, Point2f quad2_center, float feature_angle) {
    float angle_max = 0, angle_min = 360;
    int pos_quad1 = -1, pos_quad2 = -1;
    Fea.corners.clear();
    Fea.feature_angle = feature_angle;
    for (int i = 0; i < 4; i++) {
        angle_quad1[i] = atan2(quad1_center.y - quad1[i].y, quad1_center.x - quad1[i].x) * 180 / CV_PI;
        angle_quad2[i] = atan2(quad2_center.y - quad2[i].y, quad2_center.x - quad2[i].x) * 180 / CV_PI;
    }
    for (int i = 0; i < 4; i++) {
        if (min(360 - abs(angle_quad1[(i + 2) % 4] - feature_angle), abs(angle_quad1[(i + 2) % 4] - feature_angle)) + min(360 - abs(angle_quad1[(i + 3) % 4] - feature_angle), abs(angle_quad1[(i + 3) % 4] - feature_angle)) < angle_min) {
            angle_min = min(360 - abs(angle_quad1[(i + 2) % 4] - feature_angle), abs(angle_quad1[(i + 2) % 4] - feature_angle)) + min(360 - abs(angle_quad1[(i + 3) % 4] - feature_angle), abs(angle_quad1[(i + 3) % 4] - feature_angle));
            pos_quad1 = i;
        }
        if (min(360 - abs(angle_quad2[(i + 2) % 4] - feature_angle), abs(angle_quad2[(i + 2) % 4] - feature_angle)) + min(360 - abs(angle_quad2[(i + 3) % 4] - feature_angle), abs(angle_quad2[(i + 3) % 4] - feature_angle)) > angle_max) {
            angle_max = min(360 - abs(angle_quad2[(i + 2) % 4] - feature_angle), abs(angle_quad2[(i + 2) % 4] - feature_angle)) + min(360 - abs(angle_quad2[(i + 3) % 4] - feature_angle), abs(angle_quad2[(i + 3) % 4] - feature_angle));
            pos_quad2 = i;
        }
    }
    for (int i = 0; i < 4; i++) {
        Fea.corners.push_back(quad1[(i + pos_quad1) % 4]);
    }
    for (int i = 0; i < 4; i++) {
        Fea.corners.push_back(quad2[(i + pos_quad2) % 4]);
    }
    Fea.feature_center = Point2f((Fea.corners[0].x + Fea.corners[1].x + Fea.corners[4].x + Fea.corners[5].x) / 4, (Fea.corners[0].y + Fea.corners[1].y + Fea.corners[4].y + Fea.corners[5].y) / 4);
    return Fea;
}

void corner_detector::edgeRefine(const Mat& src, vector<featureInfo>& features, vector<featureInfo>& features_refined, int subPixWindow)
{
    for (int i = 0; i < features.size(); i++) {
        double lines_next[4][4], lines_last[4][4];

        for (int edge = 0; edge < 4; edge++) {
            int a = edge, b = (edge + 1) & 3;

            // compute the normal to the current line estimate
            double nx = features[i].corners[b].y - features[i].corners[a].y;
            double ny = -features[i].corners[b].x + features[i].corners[a].x;
            double mag = sqrt(nx * nx + ny * ny);
            nx /= mag;
            ny /= mag;

            int nsamples = max(128.0, mag / 8);
            double Mx = 0, My = 0, Mxx = 0, Mxy = 0, Myy = 0, N = 0;

            for (int s = 0; s < nsamples * 1; s++) {
                double alpha = (15.0 + s) / (nsamples + 30);
                double x0 = alpha * features[i].corners[a].x + (1 - alpha) * features[i].corners[b].x;
                double y0 = alpha * features[i].corners[a].y + (1 - alpha) * features[i].corners[b].y;

                double Mn = 0;
                double Mcount = 0;
                double range = subPixWindow;

                for (double n = -range; n <= range; n += 0.25) {
                    double grange = 1;
                    int x1 = x0 + (n + grange) * nx;
                    int y1 = y0 + (n + grange) * ny;
                    if (x1 < 0 || x1 >= src.cols || y1 < 0 || y1 >= src.rows)
                        continue;

                    int x2 = x0 + (n - grange) * nx;
                    int y2 = y0 + (n - grange) * ny;
                    if (x2 < 0 || x2 >= src.cols || y2 < 0 || y2 >= src.rows)
                        continue;

                    float g1 = src.at<float>(y1, x1);
                    float g2 = src.at<float>(y2, x2);

                    if (g1 < g2)
                        continue;

                    double weight = (g2 - g1) * (g2 - g1);

                    Mn += weight * n;
                    Mcount += weight;
                }

                if (Mcount == 0)
                    continue;

                double n0 = Mn / Mcount;

                double bestx = x0 + n0 * nx;
                double besty = y0 + n0 * ny;

                Mx += bestx * (1 - alpha);
                My += besty * (1 - alpha);
                Mxx += bestx * bestx * (1 - alpha);
                Mxy += bestx * besty * (1 - alpha);
                Myy += besty * besty * (1 - alpha);
                N += (1 - alpha);
            }

            double Ex = Mx / N, Ey = My / N;
            double Cxx = Mxx / N - Ex * Ex;
            double Cxy = Mxy / N - Ex * Ey;
            double Cyy = Myy / N - Ey * Ey;

            double normal_theta = .5 * atan2f(-2 * Cxy, (Cyy - Cxx));
            nx = cosf(normal_theta);
            ny = sinf(normal_theta);
            lines_next[edge][0] = Ex;
            lines_next[edge][1] = Ey;
            lines_next[edge][2] = nx;
            lines_next[edge][3] = ny;
        }

        for (int edge = 0; edge < 4; edge++) {
            int a = edge, b = (edge + 1) & 3;

            double nx = features[i].corners[b].y - features[i].corners[a].y;
            double ny = -features[i].corners[b].x + features[i].corners[a].x;
            double mag = sqrt(nx * nx + ny * ny);
            nx /= mag;
            ny /= mag;

            int nsamples = max(128.0, mag / 8);
            double Mx = 0, My = 0, Mxx = 0, Mxy = 0, Myy = 0, N = 0;

            for (int s = nsamples * 0; s < nsamples * 1; s++) {
                double alpha = (15.0 + s) / (nsamples + 30);
                double x0 = alpha * features[i].corners[a].x + (1 - alpha) * features[i].corners[b].x;
                double y0 = alpha * features[i].corners[a].y + (1 - alpha) * features[i].corners[b].y;

                double Mn = 0;
                double Mcount = 0;
                double range = subPixWindow;

                for (double n = -range; n <= range; n += 0.25) {
                    double grange = 1;
                    int x1 = x0 + (n + grange) * nx;
                    int y1 = y0 + (n + grange) * ny;
                    if (x1 < 0 || x1 >= src.cols || y1 < 0 || y1 >= src.rows)
                        continue;

                    int x2 = x0 + (n - grange) * nx;
                    int y2 = y0 + (n - grange) * ny;
                    if (x2 < 0 || x2 >= src.cols || y2 < 0 || y2 >= src.rows)
                        continue;

                    float g1 = src.at<float>(y1, x1);
                    float g2 = src.at<float>(y2, x2);

                    if (g1 < g2)
                        continue;

                    double weight = (g2 - g1) * (g2 - g1);

                    Mn += weight * n;
                    Mcount += weight;
                }

                if (Mcount == 0)
                    continue;

                double n0 = Mn / Mcount;

                double bestx = x0 + n0 * nx;
                double besty = y0 + n0 * ny;

                Mx += bestx * alpha;
                My += besty * alpha;
                Mxx += bestx * bestx * alpha;
                Mxy += bestx * besty * alpha;
                Myy += besty * besty * alpha;
                N += alpha;
            }

            // fit a line
            double Ex = Mx / N, Ey = My / N;
            double Cxx = Mxx / N - Ex * Ex;
            double Cxy = Mxy / N - Ex * Ey;
            double Cyy = Myy / N - Ey * Ey;

            double normal_theta = .5 * atan2f(-2 * Cxy, (Cyy - Cxx));
            nx = cosf(normal_theta);
            ny = sinf(normal_theta);
            lines_last[edge][0] = Ex;
            lines_last[edge][1] = Ey;
            lines_last[edge][2] = nx;
            lines_last[edge][3] = ny;
        }

        for (int it = 0; it < 4; it++) {
            double A00 = lines_next[it][3], A01 = -lines_last[(it + 1) & 3][3];
            double A10 = -lines_next[it][2], A11 = lines_last[(it + 1) & 3][2];
            double B0 = -lines_next[it][0] + lines_last[(it + 1) & 3][0];
            double B1 = -lines_next[it][1] + lines_last[(it + 1) & 3][1];

            double det = A00 * A11 - A10 * A01;
            double W00 = A11 / det, W01 = -A01 / det;
            double L0 = W00 * B0 + W01 * B1;

            double dist_old_new = distance_2points(features[i].corners[(it + 1) & 3], Point2f(lines_next[it][0] + L0 * A00, lines_next[it][1] + L0 * A10));

            if (fabs(det) > 0.001) {
                features_refined[i].corners[(it + 1) & 3].x = lines_next[it][0] + L0 * A00;
                features_refined[i].corners[(it + 1) & 3].y = lines_next[it][1] + L0 * A10;
            }
            else {
                features_refined[i].corners[(it + 1) & 3] = features[i].corners[(it + 1) & 3];
            }
        }

        for (int edge = 0; edge < 4; edge++) {
            int a = edge + 4, b = (edge + 1) % 4 + 4;

            double nx = features[i].corners[b].y - features[i].corners[a].y;
            double ny = -features[i].corners[b].x + features[i].corners[a].x;
            double mag = sqrt(nx * nx + ny * ny);
            nx /= mag;
            ny /= mag;

            int nsamples = max(128.0, mag / 8);
            double Mx = 0, My = 0, Mxx = 0, Mxy = 0, Myy = 0, N = 0;

            for (int s = 0; s < nsamples * 1; s++) {
                double alpha = (15.0 + s) / (nsamples + 30);
                double x0 = alpha * features[i].corners[a].x + (1 - alpha) * features[i].corners[b].x;
                double y0 = alpha * features[i].corners[a].y + (1 - alpha) * features[i].corners[b].y;

                double Mn = 0;
                double Mcount = 0;

                double range = subPixWindow;

                for (double n = -range; n <= range; n += 0.25) {
                    double grange = 1;
                    int x1 = x0 + (n + grange) * nx;
                    int y1 = y0 + (n + grange) * ny;
                    if (x1 < 0 || x1 >= src.cols || y1 < 0 || y1 >= src.rows)
                        continue;

                    int x2 = x0 + (n - grange) * nx;
                    int y2 = y0 + (n - grange) * ny;
                    if (x2 < 0 || x2 >= src.cols || y2 < 0 || y2 >= src.rows)
                        continue;

                    float g1 = src.at<float>(y1, x1);
                    float g2 = src.at<float>(y2, x2);

                    if (g1 < g2)
                        continue;

                    double weight = (g2 - g1) * (g2 - g1);

                    Mn += weight * n;
                    Mcount += weight;
                }

                if (Mcount == 0)
                    continue;

                double n0 = Mn / Mcount;

                double bestx = x0 + n0 * nx;
                double besty = y0 + n0 * ny;

                Mx += bestx * (1 - alpha);
                My += besty * (1 - alpha);
                Mxx += bestx * bestx * (1 - alpha);
                Mxy += bestx * besty * (1 - alpha);
                Myy += besty * besty * (1 - alpha);
                N += (1 - alpha);
            }

            double Ex = Mx / N, Ey = My / N;
            double Cxx = Mxx / N - Ex * Ex;
            double Cxy = Mxy / N - Ex * Ey;
            double Cyy = Myy / N - Ey * Ey;

            double normal_theta = .5 * atan2f(-2 * Cxy, (Cyy - Cxx));
            nx = cosf(normal_theta);
            ny = sinf(normal_theta);
            lines_next[edge][0] = Ex;
            lines_next[edge][1] = Ey;
            lines_next[edge][2] = nx;
            lines_next[edge][3] = ny;
        }

        for (int edge = 0; edge < 4; edge++) {
            int a = edge + 4, b = (edge + 1) % 4 + 4;

            double nx = features[i].corners[b].y - features[i].corners[a].y;
            double ny = -features[i].corners[b].x + features[i].corners[a].x;
            double mag = sqrt(nx * nx + ny * ny);
            nx /= mag;
            ny /= mag;

            int nsamples = max(128.0, mag / 8);
            double Mx = 0, My = 0, Mxx = 0, Mxy = 0, Myy = 0, N = 0;

            for (int s = 0; s < nsamples * 1; s++) {
                double alpha = (15.0 + s) / (nsamples + 30);
                double x0 = alpha * features[i].corners[a].x + (1 - alpha) * features[i].corners[b].x;
                double y0 = alpha * features[i].corners[a].y + (1 - alpha) * features[i].corners[b].y;

                double Mn = 0;
                double Mcount = 0;

                double range = subPixWindow;

                for (double n = -range; n <= range; n += 0.25) {
                    double grange = 1;
                    int x1 = x0 + (n + grange) * nx;
                    int y1 = y0 + (n + grange) * ny;
                    if (x1 < 0 || x1 >= src.cols || y1 < 0 || y1 >= src.rows)
                        continue;

                    int x2 = x0 + (n - grange) * nx;
                    int y2 = y0 + (n - grange) * ny;
                    if (x2 < 0 || x2 >= src.cols || y2 < 0 || y2 >= src.rows)
                        continue;

                    float g1 = src.at<float>(y1, x1);
                    float g2 = src.at<float>(y2, x2);

                    if (g1 < g2)
                        continue;

                    double weight = (g2 - g1) * (g2 - g1);

                    Mn += weight * n;
                    Mcount += weight;
                }

                if (Mcount == 0)
                    continue;

                double n0 = Mn / Mcount;

                double bestx = x0 + n0 * nx;
                double besty = y0 + n0 * ny;

                Mx += bestx * alpha;
                My += besty * alpha;
                Mxx += bestx * bestx * alpha;
                Mxy += bestx * besty * alpha;
                Myy += besty * besty * alpha;
                N += alpha;
            }

            double Ex = Mx / N, Ey = My / N;
            double Cxx = Mxx / N - Ex * Ex;
            double Cxy = Mxy / N - Ex * Ey;
            double Cyy = Myy / N - Ey * Ey;

            double normal_theta = .5 * atan2f(-2 * Cxy, (Cyy - Cxx));
            nx = cosf(normal_theta);
            ny = sinf(normal_theta);
            lines_last[edge][0] = Ex;
            lines_last[edge][1] = Ey;
            lines_last[edge][2] = nx;
            lines_last[edge][3] = ny;
        }

        for (int it = 0; it < 4; it++) {
            double A00 = lines_next[it][3], A01 = -lines_last[(it + 1) & 3][3];
            double A10 = -lines_next[it][2], A11 = lines_last[(it + 1) & 3][2];
            double B0 = -lines_next[it][0] + lines_last[(it + 1) & 3][0];
            double B1 = -lines_next[it][1] + lines_last[(it + 1) & 3][1];

            double det = A00 * A11 - A10 * A01;
            double W00 = A11 / det, W01 = -A01 / det;
            double L0 = W00 * B0 + W01 * B1;

            double dist_old_new = distance_2points(features[i].corners[(it + 1) % 4 + 4], Point2f(lines_next[it][0] + L0 * A00, lines_next[it][1] + L0 * A10));

            if (fabs(det) > 0.001) {
                features_refined[i].corners[(it + 1) % 4 + 4].x = lines_next[it][0] + L0 * A00;
                features_refined[i].corners[(it + 1) % 4 + 4].y = lines_next[it][1] + L0 * A10;
            }
            else {
                features_refined[i].corners[(it + 1) % 4 + 4] = features[i].corners[(it + 1) % 4 + 4];
            }
        }
    }
}

template <typename T, typename Compare>
std::vector<std::size_t> sort_permutation(
    const std::vector<T> vec,
    Compare compare)
{
    std::vector<std::size_t> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(),
        [&](std::size_t i, std::size_t j) { return compare(vec[i], vec[j]); });
    return p;
}

template <typename T>
std::vector<T> apply_permutation(
    const std::vector<T>& vec,
    const std::vector<std::size_t>& p)
{
    std::vector<T> sorted_vec(vec.size());
    std::transform(p.begin(), p.end(), sorted_vec.begin(),
        [&](std::size_t i) { return vec[i]; });
    return sorted_vec;
}

void corner_detector::markerOrganization(vector<featureInfo> feature, vector<MarkerInfo>& markers){
    markers.clear();
    if (feature.empty()) return;
    father.resize(feature.size());
    for (int i = 0; i < feature.size(); i++){
        father[i] = i;
    }
    for (int i = 0; i < feature.size() - 1; i++){
        for (int j = i + 1; j < feature.size(); j++){
            vector_center = feature[i].feature_center - feature[j].feature_center;
            vector_longedge = feature[i].corners[0] - feature[i].corners[5];
            center_angle = (vector_center.x * vector_longedge.x + vector_center.y * vector_longedge.y) / sqrt((vector_center.x * vector_center.x + vector_center.y * vector_center.y) * (vector_longedge.x * vector_longedge.x + vector_longedge.y * vector_longedge.y));
            if ((abs(feature[i].feature_angle - feature[j].feature_angle) < threshold_angle * 2 || abs(180 - abs(feature[i].feature_angle - feature[j].feature_angle)) < threshold_angle) && (distance_2points(feature[i].feature_center, feature[j].feature_center) < 0.3 * distance_2points(feature[i].corners[0], feature[i].corners[5])) && (abs(center_angle) < threshold_vertical)){
                int father_i = union_find(i), father_j = union_find(j);
                if (father_i != father_j)
                    father[father_j] = father_i;
            }
        }
    }

    father_database.clear();
    cnt = 0;
    vector<vector<int>> marker_ID(feature.size());
    father_database.push_back(father[0]);
    marker_ID[cnt++].push_back(0);

    for (int i = 1; i < feature.size(); i++) {
        int father_now = father[i];
        while (father_now != father[father_now])
            father_now = union_find(father_now);
        father[i] = father_now;
    }

    for (int i = 1; i < feature.size(); i++){
        find_father = false;
        for (int j = 0; j < father_database.size(); j++){
            if (father[i] == father_database[j]){
                marker_ID[j].push_back(i);
                find_father = true;
                break;
            }
        }
        if (!find_father){
            father_database.push_back(father[i]);
            marker_ID[cnt++].push_back(i);
        }
    }
    markers.clear();
    for (int i = 0; i < cnt; i++){
        MarkerInfo mark;
        marker_angle = 0;
        for (int j = 0; j < marker_ID[i].size(); j++){
            mark.cornerLists.push_back(feature[marker_ID[i][j]].corners);
            mark.feature_center.push_back(feature[marker_ID[i][j]].feature_center);
            mark.edge_length.push_back((distance_2points(feature[marker_ID[i][j]].corners[0], feature[marker_ID[i][j]].corners[1]) + distance_2points(feature[marker_ID[i][j]].corners[4], feature[marker_ID[i][j]].corners[5])) / 2);
            double angle_now = fastAtan2(mark.cornerLists[j][0].y - mark.cornerLists[j][5].y, mark.cornerLists[j][0].x - mark.cornerLists[j][5].x);
            if (angle_now > 180) angle_now -= 180;
            marker_angle += angle_now;
        }
        marker_angle /= mark.cornerLists.size();

        if (abs(marker_angle) < 45 || abs(marker_angle) > 135) {
            auto p = sort_permutation(mark.feature_center,
                [](const auto& a, const auto& b) { return a.y > b.y; });
            mark.feature_center = apply_permutation(mark.feature_center, p);
            mark.cornerLists = apply_permutation(mark.cornerLists, p);
            mark.edge_length = apply_permutation(mark.edge_length, p);
            featureExtraction(mark, mark, 0);
        }
        else {
            auto p = sort_permutation(mark.feature_center,
                [](const auto& a, const auto& b) { return a.x < b.x; });
            mark.feature_center = apply_permutation(mark.feature_center, p);
            mark.cornerLists = apply_permutation(mark.cornerLists, p);
            mark.edge_length = apply_permutation(mark.edge_length, p);
            featureExtraction(mark, mark, 1);
        }
        markers.push_back(mark);
    }
}

void corner_detector::featureExtraction(MarkerInfo& marker_src, MarkerInfo& marker_dst, int direction) {
    marker_dst = marker_src;
    marker_dst.feature_ID.clear();
    marker_dst.feature_ID_left.clear();
    marker_dst.feature_ID_right.clear();
    marker_dst.cr_left.clear();
    marker_dst.cr_right.clear();
    for (auto& points : marker_dst.cornerLists) {
        // Preserve the public corner ordering expected by the pose estimator.
        if (!direction && points[0].x > points[4].x) {
            for (int k=0; k<4; ++k) swap(points[k], points[k+4]);
        }
        const float gapLeft = norm(points[3]-points[6]);
        const float gapRight = norm(points[2]-points[7]);
        const float heightLeft = norm(points[0]-points[5]);
        const float heightRight = norm(points[1]-points[4]);
        const float crLeft = (norm(points[0]-points[6])*norm(points[3]-points[5])) /
                             std::max(1e-6f,gapLeft*heightLeft);
        const float crRight = (norm(points[1]-points[7])*norm(points[2]-points[4])) /
                              std::max(1e-6f,gapRight*heightRight);
        marker_dst.cr_left.push_back(crLeft);
        marker_dst.cr_right.push_back(crRight);
        // Rectification avoids explicitly intersecting nearly parallel lines
        // to infer which half contains the white gap (the old right-hand side
        // also incorrectly used the LEFT midpoint).
        const vector<Point2f> outside = {points[0],points[1],points[4],points[5]};
        const vector<Point2f> square = {{0,0},{1,0},{1,1},{0,1}};
        const Mat homography = getPerspectiveTransform(outside,square);
        vector<Point2f> inside;
        perspectiveTransform(vector<Point2f>{points[3],points[6],points[2],points[7]},inside,homography);
        int left=-1, right=-1;
        auto symbol = [&](float cr, bool lower) {
            if (!std::isfinite(cr) || cr<1.30f || cr>1.85f) return -1;
            int nearest=0;
            for (int j=1;j<4;++j)
                if(abs(cr-ID_cr_correspond[j])<abs(cr-ID_cr_correspond[nearest])) nearest=j;
            return lower ? 7-nearest : nearest;
        };
        // Pixel uncertainty matters for a 10 px gap; a pure percentage gate
        // rejected legitimate columns after quantization of the two edges.
        const bool geometry = gapLeft>2 && gapRight>2 &&
            abs(gapLeft-gapRight) <= 2.0f + .10f*(gapLeft+gapRight) &&
            inside.size()==4 && std::all_of(inside.begin(),inside.end(),[](Point2f q){
                return std::isfinite(q.x)&&std::isfinite(q.y)&&q.y>-.05&&q.y<1.05;
            });
        if (geometry) {
            left=symbol(crLeft,inside[0].y+inside[1].y>1);
            right=symbol(crRight,inside[2].y+inside[3].y>1);
        }
        marker_dst.feature_ID_left.push_back(left);
        marker_dst.feature_ID_right.push_back(right);
        marker_dst.feature_ID.push_back(left>=0&&right>=0 ? left*8+right : -2);
    }
}

void corner_detector::markerDecoder(vector<MarkerInfo> markers_src, vector<MarkerInfo>& markers_dst, Mat1i& state, int featureSize){
    markers_dst.clear();
    struct Hypothesis {
        double error;
        int id, start, direction, support;
        bool mirror;
    };
    constexpr float crValues[8]={1.47f,1.54f,1.61f,1.68f,1.68f,1.61f,1.54f,1.47f};
    for (const auto& group : markers_src) {
        const int count=group.cornerLists.size();
        // A single matched column is not an ID. Require multiple independent
        // columns, and compare against EVERY ID, phase and orientation.
        if(count<std::max(4,featureSize)) continue;
        vector<int> offsets(count,0);
        bool validSpacing=true;
        for(int k=1;k<count;++k) {
            const float pitch=.75f*(group.edge_length[k]+group.edge_length[k-1]);
            const float ratio=norm(group.feature_center[k]-group.feature_center[k-1])/std::max(pitch,1e-6f);
            const int step=std::max(1,int(std::round(ratio)));
            if(!std::isfinite(ratio)||ratio<.35f||step>3) { validSpacing=false; break; }
            offsets[k]=offsets[k-1]+step;
        }
        if(!validSpacing||offsets.back()>=state.cols) continue;
        vector<Hypothesis> hypotheses;
        for(int id=0;id<state.rows;++id) for(int phase=0;phase<state.cols;++phase)
        for(int direction : {1,-1}) for(bool mirror : {false,true}) {
            vector<double> residuals;
            int support=0;
            for(int k=0;k<count;++k) {
                const int left=group.feature_ID_left[k],right=group.feature_ID_right[k];
                if(left<0||right<0) continue;
                const int pos=(phase+direction*offsets[k]+state.cols)%state.cols;
                int code=state(id,pos);
                if(mirror) code=63-code;
                if(direction<0) code=(7-code%8)*8+(7-code/8);
                const int a=code/8,b=code%8;
                const double dl=group.cr_left[k]-crValues[a];
                const double dr=group.cr_right[k]-crValues[b];
                const bool sameHalf=(a>=4)==(left>=4)&&(b>=4)==(right>=4);
                const double cost=(dl*dl+dr*dr)/2 + (sameHalf ? 0.0 : .25);
                residuals.push_back(cost);
                support += sameHalf && abs(dl)<.09 && abs(dr)<.09;
            }
            if(residuals.size()<4) continue;
            std::sort(residuals.begin(),residuals.end());
            // Bound one damaged column's influence, but retain a penalty.
            // Dropping it entirely makes a wrong 4-of-5 hypothesis tie an
            // exact 5-of-5 match, discarding useful dictionary evidence.
            if(residuals.size()>=5) residuals.back()=std::min(.01,residuals.back())*.25;
            const double error=std::sqrt(std::accumulate(residuals.begin(),residuals.end(),0.0)/residuals.size());
            hypotheses.push_back({error,id,phase,direction,support,mirror});
        }
        if(hypotheses.size()<2) continue;
        std::sort(hypotheses.begin(),hypotheses.end(),[](const auto&a,const auto&b){return a.error<b.error;});
        const auto& best=hypotheses[0];
        const double margin=hypotheses[1].error-best.error;
        if(best.support<std::max(4,featureSize)||best.error>.040||margin<.012) continue;
        MarkerInfo result=group;
        result.markerID=best.id;
        result.mirrored=best.mirror;
        result.decodeError=best.error;
        result.decodeMargin=margin;
        result.supportColumns=best.support;
        result.featurePos.clear();
        for(int k=0;k<count;++k) {
            result.featurePos.push_back((best.start+best.direction*offsets[k]+state.cols)%state.cols);
            // Reflection flags allow the corner order to be mapped back to
            // canonical model coordinates without inventing a camera pose.
            if(best.mirror) {
                const auto previous=result.cornerLists[k];
                const int reflection[8]={5,4,7,6,1,0,3,2};
                for(int j=0;j<8;++j) result.cornerLists[k][j]=previous[reflection[j]];
            }
            if(best.direction<0) {
                for(int j=0;j<4;++j) swap(result.cornerLists[k][j],result.cornerLists[k][j+4]);
            }
        }
        markers_dst.push_back(result);
    }
}

inline float corner_detector::distance_2points(Point2f point1, Point2f point2){
    return sqrt((point1.x - point2.x) * (point1.x - point2.x) + (point1.y - point2.y) * (point1.y - point2.y));
}

int corner_detector::union_find(int x){
    return x == father[x] ? x : (father[x] = union_find(father[x]));
}

float corner_detector::area(featureInfo feature){
    float feature_area = 0;
    for (int i = 0; i < 4; i++){
        feature_area += feature.corners[pose[i]].x * feature.corners[pose[(i + 1) % 4]].y - feature.corners[pose[i]].y * feature.corners[pose[(i + 1) % 4]].x;
    }
    feature_area /= 2;
    return abs(feature_area);
}
