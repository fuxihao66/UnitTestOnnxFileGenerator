from __future__ import print_function

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2 as cv
mdevice = "cuda:0"

import sys, os
import common

TRT_LOGGER = trt.Logger()


def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder:
            builder.fp16_mode = True
            with builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
                builder.max_workspace_size = 1 << 30 # 256MiB
                #builder.max_batch_size = 1
                # Parse model file
                if not os.path.exists(onnx_file_path):
                    print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                    exit(0)
                print('Loading ONNX file from path {}...'.format(onnx_file_path))
                with open(onnx_file_path, 'rb') as model:
                    print('Beginning ONNX file parsing')
                    if not parser.parse(model.read()):
                        print ('ERROR: Failed to parse the ONNX file.')
                        for error in range(parser.num_errors):
                            print (parser.get_error(error))
                        return None
                # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
                print(network.get_input(0).shape)
                print('Completed parsing of ONNX file')
                print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
                engine = builder.build_cuda_engine(network)
                print("Completed creating Engine")
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
                return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()
def load_engine(engine_file_path):
    # If a serialized engine exists, use it instead of building an engine.
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
def Tensor2NP(t):
    res=t.squeeze(0).cpu().numpy().transpose([1,2,0])
    res=cv.cvtColor(res,cv.COLOR_RGB2BGR)
    res = DeToneSimple(res)
    return res
def NP2NP(t):
    res=t[0].transpose([1,2,0])
    res=cv.cvtColor(res,cv.COLOR_RGB2BGR)
    res = DeToneSimple(res)
    return res
from Small_UnetGated import UNetLWGated_FULL
import torch
# model = UNetLWGated_FULL(18, 3)
# model.load_state_dict(torch.load("totalModel.140.pth.tar")["state_dict"])
# model = model.to(mdevice)
# model.eval()

with get_engine("./optimized_model.onnx","./trt_fp16.pb") as engine, engine.create_execution_context() as context:
    with torch.no_grad():
        input = np.zeros((3, 224, 224))
        input = torch.tensor(input).unsqueeze(0).to(mdevice)
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)

        inputs[0].host = input.cpu().numpy().copy()
        prevTime = time.time()
        [output] = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        currTime = time.time()
        print("time cost {}s".format(currTime - prevTime))
        output=output.reshape(1,3,224,224)

        # gt = model(input,features,finalMask,hisBuffer)
        # print(torch.mean(torch.abs(gt-torch.from_numpy(output).to(mdevice))))

        # cv.imwrite("./resTrain/res train %s.exr" % ("gt"), Tensor2NP(gt))