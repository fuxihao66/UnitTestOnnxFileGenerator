from onnxmltools.utils.float16_converter import convert_float_to_float16
from onnx import load_model
import onnx

# onnx_model = load_model('D:/UGit/UnitTestOnnxFileGenerator/Gather-7.onnx')
# new_onnx_model = convert_float_to_float16(onnx_model)
# onnx.save(new_onnx_model, 'D:/UGit/UnitTestOnnxFileGenerator/GatherFp16-7.onnx')

onnx_model = load_model('D:/UGit/ExtraNetTRTInference-main/UNetGated_UEIntegrate.onnx')
new_onnx_model = convert_float_to_float16(onnx_model)
onnx.save(new_onnx_model, 'D:/UGit/ExtraNetTRTInference-main/UNetGated_UEIntegrate_fp16.onnx')

# onnx_model = load_model('D:/UGit/OnnxDMLPlugin/OnnxDMLTest/model/candy-9.onnx')
# new_onnx_model = convert_float_to_float16(onnx_model)
# onnx.save(new_onnx_model, 'D:/UGit/OnnxDMLPlugin/OnnxDMLTest/model/candy-fp16-9.onnx')