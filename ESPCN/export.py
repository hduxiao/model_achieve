# Some standard imports
import io
import numpy as np
import onnx
import onnxruntime
from torch import nn
import torch.onnx


from train import ESPCN

torch_model = ESPCN(upscale_factor=3)

batch_size = 1    # just a random number

model_path = './assets/models/best.pth'

# Initialize model with the pretrained weights
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
torch_model.load_state_dict(torch.load(model_path, map_location=map_location))

# set the model to inference mode
torch_model.eval()

# Input to the model
x = torch.randn(1, 1, 234, 234, requires_grad=True)
torch_out = torch_model(x)

# Export the model
# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "espcn.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=14,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : [2, 3],    # variable length axes
                                'output' : [2, 3]})


onnx_model = onnx.load("espcn.onnx")
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession("espcn.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")