#include "rock_retinafaceoutput.h"
#include "opencv2/opencv.hpp"
#include <ostream>
#include <fstream>
#include <sys/time.h>

using namespace std;

////////////////////////////////////////////////////////////////
// 
// bounding box 后处理的工具代码
//
////////////////////////////////////////////////////////////////

// bounding box信息
struct RetinafaceBBoxRect
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    int label;   //  bounding box对应label的id
    int landmark; // bounding box对应landmark的id
};

// compute IOU
static inline float intersection_area(const RetinafaceBBoxRect& a, const RetinafaceBBoxRect& b)
{
    if (a.xmin > b.xmax || a.xmax < b.xmin || a.ymin > b.ymax || a.ymax < b.ymin)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.xmax, b.xmax) - std::max(a.xmin, b.xmin);
    float inter_height = std::min(a.ymax, b.ymax) - std::max(a.ymin, b.ymin);

    return inter_width * inter_height;
}

template <typename T>
static void qsort_descent_inplace(std::vector<T>& datas, std::vector<float>& scores, int left, int right)
{
    int i = left;
    int j = right;
    float p = scores[(left + right) / 2];

    while (i <= j)
    {
        while (scores[i] > p)
            i++;

        while (scores[j] < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(datas[i], datas[j]);
            std::swap(scores[i], scores[j]);

            i++;
            j--;
        }
    }

    if (left < j)
        qsort_descent_inplace(datas, scores, left, j);

    if (i < right)
        qsort_descent_inplace(datas, scores, i, right);
}

template <typename T>
static void qsort_descent_inplace(std::vector<T>& datas, std::vector<float>& scores)
{
    if (datas.empty() || scores.empty())
        return;

    qsort_descent_inplace(datas, scores, 0, scores.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<RetinafaceBBoxRect>& bboxes, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = bboxes.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        const RetinafaceBBoxRect& r = bboxes[i];

        float width = r.xmax - r.xmin;
        float height = r.ymax - r.ymin;

        areas[i] = width * height;
    }

    printf("bbox size:%d\n",n);

    for (int i = 0; i < n; i++)
    {
        const RetinafaceBBoxRect& a = bboxes[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const RetinafaceBBoxRect& b = bboxes[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}


RockRetinafaceOuput::RockRetinafaceOuput(const RetinafaceOutputParameters &parameters)
{
    m_parameters.confidence_threshold = parameters.confidence_threshold;
    m_parameters.keep_top_k           = parameters.keep_top_k;
    m_parameters.nms_threshold        = parameters.nms_threshold;
    m_parameters.nms_top_k            = parameters.nms_top_k;
    m_parameters.num_class            = parameters.num_class;
}


int RockRetinafaceOuput::forward(const std::vector<rock::Mat>& bottom_blobs, std::vector<rock::Mat> &top_blobs)
{
    struct timeval tm_before,tm_after;

    const rock::Mat& confidence      = bottom_blobs[0];
    const rock::Mat& bbox_offset     = bottom_blobs[1];
    const rock::Mat& landmark_offset = bottom_blobs[2];
    const rock::Mat& priorbox        = bottom_blobs[3];

    const int num_prior = priorbox.w / 4 ; // prior box的总数

    const float *confidence_ptr      = (const float *)confidence.data;
    const float *landmark_offset_ptr = (const float *)landmark_offset.data;
    const float *bbox_offset_ptr     = (const float *)bbox_offset.data;
    const float *priorbox_ptr        = priorbox.row(0);
    const float *variance_ptr        = priorbox.row(1);

    vector<vector<float>> landmark_points; // landmark => [num_prior,10]

    // sort and nms for each class
    int num_class = m_parameters.num_class;
    std::vector< std::vector<RetinafaceBBoxRect> > all_class_bbox_rects;
    std::vector< std::vector<float> > all_class_bbox_scores;
    all_class_bbox_rects.resize(num_class);
    all_class_bbox_scores.resize(num_class);

    printf("#######################################\n");
    printf("nms threshold:%f\n",m_parameters.nms_threshold);
    printf("nms top k:%d\n",m_parameters.nms_top_k);
    printf("classes:%d\n",m_parameters.num_class);
    printf("confidence:%f\n",m_parameters.confidence_threshold);
    // start from 1,igore backbground
    //#pragma omp parallel for
    for(int i = 1; i < num_class; i++)
    {
        // filter by confidence_threshold
        std::vector<RetinafaceBBoxRect> class_bbox_rects;
        std::vector<float> class_bbox_scores;

        for(int j = 0 ; j < num_prior; j ++)
        {
            float score = confidence_ptr[j * num_class + i];

            if (score > m_parameters.confidence_threshold)
            {
                // 计算通过阈值的框坐标
                const float* loc = bbox_offset_ptr + j * 4;
                const float* pb = priorbox_ptr + j * 4;
                const float* var = variance_ptr + j * 4;

                // CENTER_SIZE
                float pb_w = pb[2] - pb[0];
                float pb_h = pb[3] - pb[1];
                float pb_cx = (pb[0] + pb[2]) * 0.5f;
                float pb_cy = (pb[1] + pb[3]) * 0.5f;

                float bbox_cx = var[0] * loc[0] * pb_w + pb_cx;
                float bbox_cy = var[1] * loc[1] * pb_h + pb_cy;
                float bbox_w = exp(var[2] * loc[2]) * pb_w;
                float bbox_h = exp(var[3] * loc[3]) * pb_h;

                float bbox_0 = bbox_cx - bbox_w * 0.5f;
                float bbox_1 = bbox_cy - bbox_h * 0.5f;
                float bbox_2 = bbox_cx + bbox_w * 0.5f;
                float bbox_3 = bbox_cy + bbox_h * 0.5f;

                RetinafaceBBoxRect c = { bbox_0, bbox_1, bbox_2, bbox_3, i , j};
                class_bbox_rects.push_back(c);
                class_bbox_scores.push_back(score);
            }
        }

        // sort inplace
        qsort_descent_inplace(class_bbox_rects, class_bbox_scores);

        // keep nms_top_k
        if (m_parameters.nms_top_k < (int)class_bbox_rects.size())
        {
            class_bbox_rects.resize(m_parameters.nms_top_k);
            class_bbox_scores.resize(m_parameters.nms_top_k);
        }

        // apply nms
        std::vector<int> picked;
        nms_sorted_bboxes(class_bbox_rects, picked, m_parameters.nms_threshold);

        printf("class = %d,nms picked count:%d\n",i,picked.size());

        // select
        for (int j = 0; j < (int)picked.size(); j++)
        {
            int z = picked[j];
            all_class_bbox_rects[i].push_back(class_bbox_rects[z]);
            all_class_bbox_scores[i].push_back(class_bbox_scores[z]);
        }
    }
    //gettimeofday(&tm_after,NULL);
    //printf("[DEBUG] step2 spend time : %d ms\n",(tm_after.tv_sec - tm_before.tv_sec)*1000+(tm_after.tv_usec - tm_before.tv_usec)/1000);

    // gather all class
    std::vector<RetinafaceBBoxRect> bbox_rects;
    std::vector<float> bbox_scores;

    for (int i = 1; i < num_class; i++)
    {
        const std::vector<RetinafaceBBoxRect>& class_bbox_rects = all_class_bbox_rects[i];
        const std::vector<float>& class_bbox_scores = all_class_bbox_scores[i];

        bbox_rects.insert(bbox_rects.end(), class_bbox_rects.begin(), class_bbox_rects.end());
        bbox_scores.insert(bbox_scores.end(), class_bbox_scores.begin(), class_bbox_scores.end());
    }

    printf("bbox rects size:%d\n",bbox_rects.size());
    printf("bbox score size:%d\n",bbox_scores.size());

    if(bbox_scores.empty() && bbox_rects.empty())
    {
        printf("empty,return\n");
        return -1;
    }

    // global sort inplace
    qsort_descent_inplace(bbox_rects, bbox_scores);

    // keep_top_k
    if (m_parameters.keep_top_k < (int)bbox_rects.size())
    {
        bbox_rects.resize(m_parameters.keep_top_k);
        bbox_scores.resize(m_parameters.keep_top_k);
    }

    int num_detected = bbox_rects.size();

    // 填充bounding box 数据到 Mat中返回
    rock::Mat bounding_box_blob;
    bounding_box_blob.create(6,num_detected);

    if (bounding_box_blob.empty())
    {
        return -100;
    }

    for (int i = 0; i < num_detected; i++)
    {
        const RetinafaceBBoxRect& r = bbox_rects[i];
        float score = bbox_scores[i];
        float* output_ptr = bounding_box_blob.row(i);

        output_ptr[0] = r.label;
        output_ptr[1] = score;
        output_ptr[2] = r.xmin;
        output_ptr[3] = r.ymin;
        output_ptr[4] = r.xmax;
        output_ptr[5] = r.ymax;
    }

    // 填充landmark数据到Mat中返回
    rock::Mat landmark_blob;
    landmark_blob.create(10,num_detected);

    if(landmark_blob.empty())
    {
        return -100;
    }
        
    for(int i = 0; i < num_detected; i ++)
    {
        const RetinafaceBBoxRect& r = bbox_rects[i];
        float* output_ptr = landmark_blob.row(i);

        int landmark_index = r.landmark * 10;
        int index = r.landmark * 4;

        float pb_w  = priorbox_ptr[index + 2] - priorbox_ptr[index];
        float pb_h  = priorbox_ptr[index + 3] - priorbox_ptr[index + 1];
        float pb_cx = (priorbox_ptr[index] + priorbox_ptr[index + 2]) * 0.5f;
        float pb_cy = (priorbox_ptr[index + 3] + priorbox_ptr[index + 1]) *0.5f;

        for(int k = 0; k < 5; k ++)
        {
            int prior_land_index = k * 2;

            float land_x = landmark_offset_ptr[landmark_index + prior_land_index];
            float land_y = landmark_offset_ptr[landmark_index + prior_land_index + 1];

            float x = pb_cx + variance_ptr[index] * land_x * pb_w;
            float y = pb_cy + variance_ptr[index + 1] * land_y * pb_h;
            
            output_ptr[2*k] = x;
            output_ptr[2*k+1] = y;
        }

    }

    top_blobs.push_back(bounding_box_blob);
    top_blobs.push_back(landmark_blob);

    return 0;
}
