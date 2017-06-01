#!/usr/bin/python3

import pyopencl as cl
import numpy
import sys
import math

from PIL import Image
numpy.set_printoptions(threshold=numpy.nan) # print all data

# TODO create kernels
kernel = """
__kernel void convolve(
    read_only image2d_t input,
    write_only image2d_t output,
    const int width,
    const int height
    )
{
    const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    uint4 pix = 4 * read_imageui(input, sampler, pos);
    pix += read_imageui(input, sampler, (int2)(pos.x - 1, pos.y - 1));
    pix += read_imageui(input, sampler, (int2)(pos.x - 1, pos.y)) * 2;
    pix += read_imageui(input, sampler, (int2)(pos.x - 1, pos.y + 1));
    pix += read_imageui(input, sampler, (int2)(pos.x , pos.y - 1)) * 2;
    pix += read_imageui(input, sampler, (int2)(pos.x , pos.y + 1)) * 2;
    pix += read_imageui(input, sampler, (int2)(pos.x + 1, pos.y - 1));
    pix += read_imageui(input, sampler, (int2)(pos.x + 1, pos.y)) * 2;
    pix += read_imageui(input, sampler, (int2)(pos.x + 1, pos.y + 1));
    //pix /= (uint4)(16, 16, 16, 16);
    pix.x /= 16;
    pix.y /= 16;
    pix.z /= 16;
    pix.w /= 16;
    write_imageui(output, pos, pix);
}
"""

# set up device
context = cl.create_some_context()
queue = cl.CommandQueue(context)

# build the kernels
program = cl.Program(context, kernel).build()

# read image and construct array
img = Image.open(sys.argv[1])
if img.mode != "RGBA":
    img = img.convert("RGBA")
img_arr = numpy.asarray(img).astype(numpy.uint8)
(img_h, img_w, bytes_per_pixel) = img_arr.shape
size = (img_w, img_h)

# image format
fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)

# input buffers
inImg = cl.Image(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, fmt, size, None, img_arr)

# output buffers
outImg = cl.Image(context, cl.mem_flags.WRITE_ONLY, fmt, size)

global_work_size = size
local_work_size = None

convolve = program.convolve
convolve.set_scalar_arg_dtypes([None, None, numpy.int32, numpy.int32])
convolve(queue, global_work_size, local_work_size, inImg, outImg, img_w, img_h)

# wait for queue to finish
queue.finish()

# read the result
buffer = numpy.zeros(size[0] * size[1] * 4, numpy.uint8)  
origin = (0, 0, 0 )  
region = (size[0], size[1], 1 )  
      
cl.enqueue_read_image(queue, outImg, origin, region, buffer).wait()  

gsim = Image.frombytes("RGBA", size, buffer.tostring())
gsim.save("out.jpg")
