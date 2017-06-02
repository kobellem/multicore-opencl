#!/usr/bin/python3

import pyopencl as cl
import numpy
import sys
import math

from PIL import Image
numpy.set_printoptions(threshold=numpy.nan) # print all data

# create masks
def normalize_mask(kernel, dim):
    """Normalizes a kernel
    Args:
        kernel: a two-d kernel
    """
    for x in range(0, dim):
        for y in range(dim):
            kernel[x][y] = kernel[x][y] / numpy.sum(kernel)
    return kernel


def gaussian_mask(dim, sigma):
    """
    The Guassian blur function is as follows:

                           x² + y²
    G(x,y) =    1        - -------
            --------- * e    2σ²
              2πσ²
    Finally the kernel is normalized to avoid too dark or too light areas.
    """
    rows = dim
    cols = dim
    arr = numpy.empty([rows, cols]).astype(numpy.float32)
    center = dim / 2
    total = 0.0
    for x in range(0, rows):
        for y in range(0, cols):
            x_ = x - center
            y_ = y - center
            arr[x][y] = (1 / (2.0 * math.pi * math.pow(sigma, 2))) * math.pow(math.e, -1.0 * (
                (math.pow(x_, 2) + math.pow(y_, 2)) / (2.0 * math.pow(sigma, 2))))
            total = total + arr[x][y]

    return normalize_kernel(arr, dim)

# the convolve kernel
kernel = """
__kernel void convolve(
    __read_only image2d_t input,
    __write_only image2d_t output,
    const int dim,
    const float sigma
    )
{
    const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    int2 pos = (int2)(get_global_id(0), get_global_id(1));

    uint4 res = read_imageui(input, sampler, pos);

    write_imageui(output, pos, res);
}"""

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

# work sizes
global_work_size = size
local_work_size = None

# run the kernel
convolve = program.convolve
convolve.set_scalar_arg_dtypes([None, None, numpy.uint32, numpy.float32])
convolve(queue, global_work_size, local_work_size, inImg, outImg, 6, 1)

# wait for queue to finish
queue.finish()

# read the result
buffer = numpy.zeros(size[0] * size[1] * 4, numpy.uint8)  
origin = (0, 0, 0 )  
region = (size[0], size[1], 1 )  

cl.enqueue_read_image(queue, outImg, origin, region, buffer).wait()  

# save output image to file
gsim = Image.frombytes("RGBA", size, buffer.tostring())
gsim.save("out.jpg")
