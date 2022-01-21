#include "rock_priorbox.h"
#include <algorithm>
#include <math.h>


RockPriorBox::RockPriorBox(PriorBoxParam &priorbox_param)
{
	min_sizes = priorbox_param.min_sizes;
    max_sizes = priorbox_param.max_sizes;
    aspect_ratios = priorbox_param.aspect_ratios;
    variances[0] = priorbox_param.variances[0];
    variances[1] = priorbox_param.variances[1];
    variances[2] = priorbox_param.variances[2];
    variances[3] = priorbox_param.variances[3];
    flip = priorbox_param.flip;
    clip = priorbox_param.clip;
    image_width = priorbox_param.image_width;
    image_height = priorbox_param.image_height;
    offset = priorbox_param.offset;
}

int RockPriorBox::forward(const int bottom_blobs_w,const int bottom_blobs_h,rock::Mat& top_blobs)
{
    int w = bottom_blobs_w;
    int h = bottom_blobs_h;

    int image_w = image_width;
    int image_h = image_height;

    float step_w = step_width;
	float step_h = step_height;
    step_w = (float)image_w / w;
    step_h = (float)image_h / h;

    int num_min_size = min_sizes.size();
    int num_max_size = max_sizes.size();
    int num_aspect_ratio = aspect_ratios.size();
	
    int num_prior = num_min_size * num_aspect_ratio + num_min_size + num_max_size;
    if (flip)
        num_prior += num_min_size * num_aspect_ratio;
	
    rock::Mat& top_blob = top_blobs;
    top_blob.create(4 * w * h * num_prior, 2);
	
    #pragma omp parallel for
    for (int i = 0; i < h; i++)
    {
        float* box = (float*)top_blob + i * w * num_prior * 4;

        float center_x = offset * step_w;
        float center_y = offset * step_h + i * step_h;

        for (int j = 0; j < w; j++)
        {
            float box_w;
            float box_h;

            for (int k = 0; k < num_min_size; k++)
            {
                float min_size = min_sizes[k];
				
                // min size box
                box_w = box_h = min_size;

                box[0] = (center_x - box_w * 0.5f) / image_w;
                box[1] = (center_y - box_h * 0.5f) / image_h;
                box[2] = (center_x + box_w * 0.5f) / image_w;
                box[3] = (center_y + box_h * 0.5f) / image_h;

                box += 4;

                if (num_max_size > 0)
                {
                    float max_size = max_sizes[k];

                    // max size box
                    box_w = box_h = sqrt(min_size * max_size);

                    box[0] = (center_x - box_w * 0.5f) / image_w;
                    box[1] = (center_y - box_h * 0.5f) / image_h;
                    box[2] = (center_x + box_w * 0.5f) / image_w;
                    box[3] = (center_y + box_h * 0.5f) / image_h;

                    box += 4;
                }

                // all aspect_ratios
                for (int p = 0; p < num_aspect_ratio; p++)
                {
                    float ar = aspect_ratios[p];

                    box_w = min_size * sqrt(ar);
                    box_h = min_size / sqrt(ar);

                    box[0] = (center_x - box_w * 0.5f) / image_w;
                    box[1] = (center_y - box_h * 0.5f) / image_h;
                    box[2] = (center_x + box_w * 0.5f) / image_w;
                    box[3] = (center_y + box_h * 0.5f) / image_h;

                    box += 4;

                    if (flip)
                    {
                        box[0] = (center_x - box_h * 0.5f) / image_w;
                        box[1] = (center_y - box_w * 0.5f) / image_h;
                        box[2] = (center_x + box_h * 0.5f) / image_w;
                        box[3] = (center_y + box_w * 0.5f) / image_h;

                        box += 4;
                    }
                }
            }

            center_x += step_w;
        }

        center_y += step_h;
    }
	
    if (clip)
    {
        float* box = top_blob;
        for (int i = 0; i < top_blob.w; i++)
        {
            box[i] = std::min(std::max(box[i], 0.f), 1.f);
        }
    }
	
    // set variance
    float* var = top_blob.row(1);
    for (int i = 0; i < top_blob.w / 4; i++)
    {
        var[0] = variances[0];
        var[1] = variances[1];
        var[2] = variances[2];
        var[3] = variances[3];

        var += 4;
    }

    return 0;
}

