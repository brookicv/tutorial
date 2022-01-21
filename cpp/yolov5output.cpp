#include <float.h>
#include <math.h>
#include <stdio.h>
#include <vector>

#include "rock_yolov5output.h"
#include "model_common_utils.h"
#include "dbg-macro-0.4.0/dbg.h"

using namespace std;


static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
    //return static_cast<float>(1.f / (1.f + RockFastExp(-x)));
}


RockYolov5DetectionOutput::RockYolov5DetectionOutput()
{
    
}

RockYolov5DetectionOutput::~RockYolov5DetectionOutput()
{

}


static inline float intersection_area(const BBoxRectYolo& a, const BBoxRectYolo& b)
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


void RockYolov5DetectionOutput::qsort_descent_inplace(std::vector<BBoxRectYolo>& datas, int left, int right) const
{
    int i = left;
    int j = right;
    float p = datas[(left + right) / 2].score;

    while (i <= j)
    {
        while (datas[i].score > p)
            i++;

        while (datas[j].score < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(datas[i], datas[j]);

            i++;
            j--;
        }
    }

    if (left < j)
        qsort_descent_inplace(datas, left, j);

    if (i < right)
        qsort_descent_inplace(datas, i, right);
}


void RockYolov5DetectionOutput::qsort_descent_inplace(std::vector<BBoxRectYolo>& datas) const
{
    if (datas.empty())
        return;

    qsort_descent_inplace(datas, 0, static_cast<int>(datas.size() - 1));
}


void RockYolov5DetectionOutput::nms_sorted_bboxes(std::vector<BBoxRectYolo>& bboxes, std::vector<size_t>& picked, float nms_threshold) const
{
    picked.clear();

    const size_t n = bboxes.size();

    for (size_t i = 0; i < n; i++)
    {
        const BBoxRectYolo& a = bboxes[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const BBoxRectYolo& b = bboxes[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = a.area + b.area - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area > nms_threshold * union_area)
            {
                keep = 0;
                break;
            }
        } 

        if (keep)
        {
            picked.push_back(i);
        }
    }
}


int RockYolov5DetectionOutput::forward(const std::vector<rock::Mat>& bottom_blobs, std::vector<rock::Mat>& top_blobs) const
{

    std::vector<std::vector<BBoxRectYolo>> all_bbox_nms_rects;
    all_bbox_nms_rects.resize(num_class-1);
    for (size_t b = 0; b < bottom_blobs.size(); b++)
    {
        const rock::Mat& bottom_top_blobs = bottom_blobs[b];

        int w = bottom_top_blobs.w;
        int h = bottom_top_blobs.h;
        int channels = bottom_top_blobs.c;

        dbg(w,h,channels);

        const int channels_per_box = channels / num_box;
        //printf("\033[;32m[DEBUG]\033[0m channel_per_box: %d, channels: %d, num_box: %d \n",channels_per_box, channels, num_box);

        if (channels_per_box != 4 + 1 + num_class - 1)
        {
            printf("\033[47;31m[ERROR]\033[0m channels_per_box != 4 + 1 + num_class - 1\n");
            return -1;
        }
        
        for (int pp = 0; pp < num_box; pp++)
        {
            int p = pp * channels_per_box;

            float anchor_w = anchors_scale[b][pp * 2];
            float anchor_h = anchors_scale[b][pp * 2 + 1];

            const float* xptr = (float*)bottom_top_blobs.channel(p).data;
            const float* yptr = (float*)bottom_top_blobs.channel(p + 1).data;
            const float* wptr = (float*)bottom_top_blobs.channel(p + 2).data;
            const float* hptr = (float*)bottom_top_blobs.channel(p + 3).data; 
            const float* box_score_ptr = bottom_top_blobs.channel(p + 4);

            // sigmod class scores
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    // find class index with max class score
                    int class_index = 0;
                    float class_score = -FLT_MAX;

                    for (int q = 0; q < num_class - 1; q++)
                    {
                        float score = bottom_top_blobs.channel(p + 5 + q).row(i)[j];
                        if (score > class_score)
                        {
                            class_index = q;
                            class_score = score;
                        }
                    }
                    
                    float conf = sigmoid(box_score_ptr[0]); //该anchor的物体得概率
                    float pred = sigmoid(class_score); //该anchor的类别概率
                    float confidence = conf * pred;

                    if (confidence >= confidence_threshold)
                    {
                        // YOLOV5的坐标解码方式跟V2V3不一样的
                        float dx = sigmoid(xptr[0]);
                        float dy = sigmoid(yptr[0]);
                        float dw = sigmoid(wptr[0]);
                        float dh = sigmoid(hptr[0]);

                        float bbox_cx = (dx * 2.f - 0.5f + j);
                        bbox_cx = bbox_cx/float(w);
                        float bbox_cy = (dy * 2.f - 0.5f + i);
                        bbox_cy = bbox_cy/float(h);

                        float bbox_w = pow(dw * 2.f, 2) * anchor_w;
                        bbox_w = bbox_w/float(model_input_w);
                        float bbox_h = pow(dh * 2.f, 2) * anchor_h;
                        bbox_h = bbox_h/float(model_input_h);

                        float bbox_xmin = bbox_cx - bbox_w * 0.5f;
                        float bbox_ymin = bbox_cy - bbox_h * 0.5f;
                        float bbox_xmax = bbox_cx + bbox_w * 0.5f;
                        float bbox_ymax = bbox_cy + bbox_h * 0.5f;

                        float area = bbox_w * bbox_h;
                        BBoxRectYolo c = {confidence, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, area, class_index};
                        all_bbox_nms_rects[class_index].emplace_back(c);
                    }

                    xptr++;
                    yptr++;
                    wptr++;
                    hptr++;

                    box_score_ptr++;
                }
            }
        }

    }
    //nms each class
    vector<BBoxRectYolo> all_bbox_nms_sel_rects;
    for(int i = 0 ; i < num_class-1; i++){

        if(all_bbox_nms_rects[i].size() > 1)
        {
            // sort inplace
            qsort_descent_inplace(all_bbox_nms_rects[i]);

            // apply nms
            std::vector<size_t> picked;
            nms_sorted_bboxes(all_bbox_nms_rects[i], picked, nms_threshold);

            // select
            for (int j = 0; j < (int)picked.size(); j++)
            {
                int z = picked[j];
                all_bbox_nms_sel_rects.emplace_back(all_bbox_nms_rects[i][z]);
            }
        }
        else if (all_bbox_nms_rects[i].size() == 1)
        {
            all_bbox_nms_sel_rects.emplace_back(all_bbox_nms_rects[i][0]);
        }
    }

    // global sort inplace
    qsort_descent_inplace(all_bbox_nms_sel_rects);
    // fill result
    int num_detected = static_cast<int>(all_bbox_nms_sel_rects.size());
    if (num_detected == 0)
    {
        return 0;
    }
    
    rock::Mat& top_blob = top_blobs[0];
    
    top_blob.create(6, num_detected, sizeof(float));
    
    if (top_blob.empty())
    {
        printf("[ERROR]top_blob.empty()\n");
        return -100;
    }
    
    for (int i = 0; i < num_detected; i++)
    {
        const BBoxRectYolo& r = all_bbox_nms_sel_rects[i];
        float score = r.score;
        float* outptr = top_blob.row(i);

        outptr[0] = static_cast<float>(r.label + 1); // +1 for prepend background class
        outptr[1] = score;
        outptr[2] = r.xmin;
        outptr[3] = r.ymin;
        outptr[4] = r.xmax;
        outptr[5] = r.ymax;
    }

    return 0;
}
