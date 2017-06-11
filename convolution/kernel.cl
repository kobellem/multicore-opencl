__kernel void convolve(
	__global float* mask,
					 int dim,
	__read_only image2d_t input,
	__write_only image2d_t output,
	__local uint4* cache
	)
{
	const sampler_t sampler =	CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
	int BLOCK_SIZE=16; //process BLOCK_SIZExBLOCK_SIZE parts of the image
	uint mid = dim>>1;

	// positions
	int2 group_pos = (get_group_id(0), get_group_id(1));
	int2 local_pos = (get_local_id(0), get_local_id(1));
	int2 global_pos = ((group_pos.x * BLOCK_SIZE + local_pos.x), (group_pos.y * BLOCK_SIZE + local_pos.y));

	// read pixels into local storage
	int2 shifted_pos = (global_pos.x - mid, global_pos.y - mid);
	if (shifted_pos.x >= 0 && shifted_pos.y >= 0) {
		uint4 pix = read_imageui(input, sampler, shifted_pos);
		cache[local_pos.x + local_pos.y * (BLOCK_SIZE + dim)] = pix;		
		barrier(CLK_LOCAL_MEM_FENCE);

		if (global_pos.x < (get_image_width(input) - mid) && global_pos.y < (get_image_height(input) - mid)) {
			float4 temp;
			float4 acc = (0.0f,0.0f,0.0f,0.0f);
			int2 current_pos;

			if (local_pos.x >= mid &&
					local_pos.x < (BLOCK_SIZE + dim - 1 - mid) &&
					local_pos.y >= mid &&
					local_pos.y < (BLOCK_SIZE + dim - 1 - mid)) {
					for (int i = 0; i < dim; i++) {
							for (int j = 0; j < dim; j++) {
									current_pos.x = local_pos.x + i - mid;
									current_pos.y = local_pos.y + j - mid;
									pix = cache[current_pos.x + current_pos.y * (BLOCK_SIZE + dim)];
									temp = (float4)((float)pix.x,
																	(float)pix.y,
																	(float)pix.z,
																	(float)pix.w); 
									acc += temp * mask[i + j * dim];
							}
					}
				printf("writing pix\n");
				write_imageui(output, shifted_pos, (uint4)((uint)acc.x,
																									 (uint)acc.y,
																									 (uint)acc.z,
																									 (uint)acc.w));
				barrier(CLK_LOCAL_MEM_FENCE);
			}
		}
	}
}
