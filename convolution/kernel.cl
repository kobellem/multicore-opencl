__kernel void convolve(
    __global float* mask,
             int dim,
    __read_only image2d_t input,
    __write_only image2d_t output
    )
{
    const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    int2 pos = (int2)(get_global_id(0), get_global_id(1));

    uint mid = dim>>1;
    float4 temp;
    uint4 pix;
    float4 acc = (0.0f,0.0f,0.0f,0.0f);
    int2 current_pos;

    if(pos.x >= mid && pos.x < (get_image_width(input) - mid) && pos.y >= mid && pos.y < (get_image_height(input) - mid)) {
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                current_pos.x = pos.x + i - mid;
                current_pos.y = pos.y + j - mid;
                pix = read_imageui(input, sampler, current_pos);
                temp = (float4)((float)pix.x, (float)pix.y, (float)pix.z, (float)pix.w);
                acc += temp * mask[i + j * dim];
            }
        }
        pix = (uint4)((uint)acc.x, (uint)acc.y, (uint)acc.z, (uint)acc.w);
    } else {
        pix = read_imageui(input, sampler, pos);
    }

    write_imageui(output, pos, pix);
}
