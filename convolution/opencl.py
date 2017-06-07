#!/usr/bin/python3

import pyopencl as cl
import numpy
import sys
import math
import time

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

    return normalize_mask(arr, dim)

def apply_kernel(mask, dim, img_buffer):
    # input buffers
    inMask = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=mask)
    inImg = cl.Image(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, fmt, size, None, img_buffer)

    # output buffers
    outImg = cl.Image(context, cl.mem_flags.WRITE_ONLY, fmt, size)

    # work sizes
    global_work_size = size
    local_work_size = None

    # run the kernel
    convolve = program.convolve
    convolve.set_scalar_arg_dtypes([None, numpy.uint32, None, None])

    start_time = time.time()
    convolve(queue, global_work_size, local_work_size, inMask, dim, inImg, outImg)

    # wait for queue to finish
    queue.finish()
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000
    print("executione time: ", elapsed_time, "ms")

    # read the result
    buffer = numpy.zeros(size[0] * size[1] * 4, numpy.uint8)  
    origin = (0, 0, 0 )  
    region = (size[0], size[1], 1 )  

    cl.enqueue_read_image(queue, outImg, origin, region, buffer).wait()  

    return buffer

def fault_tolerance(res_ocl, res_seq): 
    tolerance = 0.1
    correct = 0
    wrong = 0
    for i in range(res_seq.size):
        expected = res_seq[i]
        actual = res_ocl[i]
        absolute_error = numpy.absolute((int(actual) - int(expected)) / 255)
        if absolute_error <= tolerance:
            correct += 1
        else:
            wrong += 1

    print("correct # values ", correct)
    print("wrong # values ", wrong)
    print("total error: ", wrong / (wrong + correct))
    return correct

if __name__ == "__main__":
    # set up device
    context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    # build the kernel
    kernelsource = open("kernel.cl").read()
    program = cl.Program(context, kernelsource).build()

    # read image and construct array
    img = Image.open(sys.argv[1])
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    img_arr = numpy.asarray(img).astype(numpy.uint8)
    (img_h, img_w, bytes_per_pixel) = img_arr.shape
    size = (img_w, img_h)

    # image format
    fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)

    # create mask
    dim = int(sys.argv[2])
    sig = 1
    mask = gaussian_mask(dim, sig)
    print(mask)

    x = 10
    while x > 0:
        img_arr = apply_kernel(mask, dim, img_arr)
        x = x - 1

    # save output image to file
    gsim = Image.frombytes("RGBA", size, img_arr.tostring())
    gsim.save("out.jpg")

    # check the result with expected output
    img_seq = Image.open("output_seq.jpg")
    if img_seq.mode != "RGBA":
        img_seq = img_seq.convert("RGBA")
    img_seq_arr = numpy.asarray(img_seq).astype(numpy.uint8)
    (img_seq_h, img_seq_w, bytes_per_pixel) = img_seq_arr.shape
    flat_img_seq_arr = img_seq_arr.reshape((img_seq_h * img_seq_w * bytes_per_pixel))
    fault_tolerance(img_arr, flat_img_seq_arr)
