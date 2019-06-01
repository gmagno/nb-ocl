#!/usr/bin/env python3

import os

from jinja2 import Template
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
# import pyopencl.tools as cL_tools


def compute(
    imw=128, imh=128,
    coefs=[1, 0, 0, 0, 0, 0, 1],
    crmin=-5.0, crmax=5.0,
    cimin=-5.0, cimax=5.0,
    itmax=30, tol=1e-6,
    precision='float',  # either 'float' or 'double'
):
    coefs_t = np.float32 if precision=='float' else np.float64
    coefs = np.array(coefs, dtype=coefs_t)
    roots = np.roots(coefs)  # return a list of np.complex128
    roots = roots.astype(
        np.complex64 if precision=='float' else np.complex128
    )
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

    platform = cl.get_platforms()[0]
    # # print('platform: {}'.format(platform))
    device = platform.get_devices()[0]
    # # print('device: {}'.format(device))
    context = cl.Context([device])
    # context = cl.create_some_context()
    # print('context: {}'.format(context))
    queue = cl.CommandQueue(context)

    def cnew(r, i):
        return 'CFUNC(new)({}, {})'.format(r, i)

    def cmul(c1, c2):
        return 'CFUNC(mul)({}, {})'.format(c1, c2)

    def cmulr(c, r):
        return 'CFUNC(mulr)({}, {})'.format(c, r)

    def cadd(c1, c2):
        return 'CFUNC(add)({}, {})'.format(c1, c2)

    def caddr(c, r):
        return 'CFUNC(addr)({}, {})'.format(c, r)

    def cpolyval(p, x):
        ret = ''
        for i, c in enumerate(p):
            if i == 0:
                ret += cnew(c, 0.0)
            else:
                ret = caddr(cmul(x, ret), c)
        return ret + ';'

    def cpolyderval(p, x):
        ret = ''
        for i, c in enumerate(p):
            if i == len(p)-1:
                continue
            if i == 0:
                ret += cnew(c * (len(p)-1-i), 0.0)
            else:
                ret = cadd(cnew(c * (len(p)-1-i), 0.0), cmul(ret, x))
        return ret + ';'

    # kernel_file = 'nb.cl'
    kernel_file = 'nb.cl.j2'
    with open(kernel_file, 'r') as f:
        kernel_source = f.read()
    tpl = Template(kernel_source)
    tpl_ctx = dict(
        cpolyval_z=cpolyval(coefs, 'z'),
        cpolyderval_z=cpolyderval(coefs, 'z')
    )
    tpl_kernel_source = tpl.render(**tpl_ctx)

    crmin_t = np.float32 if precision=='float' else np.float64
    crmax_t = np.float32 if precision=='float' else np.float64
    cimin_t = np.float32 if precision=='float' else np.float64
    cimax_t = np.float32 if precision=='float' else np.float64
    tol_t   = np.float32 if precision=='float' else np.float64


    options = []
    if precision == 'double':
        options += ['-D PRECISION_DOUBLE=1']
    program = cl.Program(context, tpl_kernel_source).build(
        options=options,
    )
    program.compute.set_scalar_arg_dtypes([
        np.uint32,          # imw
        np.uint32,          # imh
        None,               # const float/double *coefs
        np.uint32,          # ncoefs
        None,               # roots
        crmin_t,            # crmin
        crmax_t,            # crmax
        cimin_t,            # cimin
        cimax_t,            # cimax
        np.uint32,          # itmax
        tol_t,              # tolerance
        None,               # __global float *h
        # None,             # __global float *s
        None,               # __global float *v
    ])

    mem_flags = cl.mem_flags
    coefs_buf = cl.Buffer(
        context,
        mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,
        hostbuf=coefs
    )
    roots_buf = cl.Buffer(
        context,
        mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,
        hostbuf=roots
    )
    h_out = cl.Buffer(
        context,
        cl.mem_flags.WRITE_ONLY,
        imw*imh*np.dtype(np.float32).itemsize
    )
    # s_out = cl.Buffer(
    #     context,
    #     cl.mem_flags.WRITE_ONLY,
    #     imw*imh*np.dtype(np.float32).itemsize
    # )
    v_out = cl.Buffer(
        context,
        cl.mem_flags.WRITE_ONLY,
        imw*imh*np.dtype(np.float32).itemsize
    )

    globalrange = (imw, imh)
    localrange = None

    program.compute(
        queue, globalrange, localrange,
        np.uint32(imw),         # const uint imw
        np.uint32(imh),         # const uint imh
        coefs_buf,              # __global float/double *coefs
        np.uint32(len(coefs)),  # const uint ncoefs
        roots_buf,              # __global cdouble_t *roots
        crmin_t(crmin),         # const float crmin
        crmax_t(crmax),         # const float crmax
        cimin_t(cimin),         # const float cimin
        cimax_t(cimax),         # const float cimax
        np.uint32(itmax),       # const uint itmax
        tol_t(tol),             # const float tolerance,
        h_out,                  # __global float *h_out
        # s_out,                # __global float *s_out
        v_out,                  # __global float *v_out
    )

    host_h = np.zeros((imh, imw), dtype=np.float32)
    host_s = np.ones((imh, imw), dtype=np.float32)
    host_v = np.zeros((imh, imw), dtype=np.float32)
    cl.enqueue_copy(queue, host_h, h_out)
    # cl.enqueue_copy(queue, host_s, s_out)
    cl.enqueue_copy(queue, host_v, v_out)
    hsv = np.dstack((host_h, host_s, host_v))

    return hsv


def main():
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

    # crmin = -10.0
    # crmax = 10.0
    # cimin = -5.0
    # cimax = 5.0
    # ratio = (crmax - crmin) / (cimax - cimin)
    # imw = 1024

    # crmin = -10.0
    # crmax = 10.0
    # cimin = -7.0
    # cimax = 7.0
    # ratio = (crmax - crmin) / (cimax - cimin)

    crmin = -10.0
    crmax = 10.0
    cimin = -5.0
    cimax = 5.0
    ratio = (crmax - crmin) / (cimax - cimin)

    # imw = 1024
    # imw = 2048
    # imw = 4096
    imw = 8192

    import time
    t = time.time()
    hsv = compute(
        imw=imw, imh=int(imw/ratio),
        coefs=[1, 0, 0, 0, 0, 0, 1],
        crmin=crmin, crmax=crmax,
        cimin=cimin, cimax=cimax,
        itmax=30, tol=1e-6,
        precision='float',
    )
    e = time.time() - t
    print('Elapsed time: %.2f' %(e,))

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    fig = plt.figure('nbm_cl')
    plt.imshow(
        mpl.colors.hsv_to_rgb(hsv),
        # cmap=plt.get_cmap('viridis')
        # interpolation='bilinear',
    )
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()
