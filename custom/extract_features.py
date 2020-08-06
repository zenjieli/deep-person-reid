import sys
import time
import os.path as osp
import argparse
import torch
import torch.nn as nn
import numpy as np

import torchreid
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)

def load_and_infer(input0):
    import onnxruntime as ort
    import onnx    
    import torchvision.transforms as transforms

    model_path = '/tmp/resnet50_output.onnx'
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    ort_session = ort.InferenceSession(model_path)
    
    outputs = ort_session.run(None, {'input0': input0})        
    test_result(outputs[0])

def test_result(np_output):
    np.testing.assert_equal(np_output.shape, (1, 2048))
    np.testing.assert_almost_equal(np_output[0][0], 0, decimal = 4, verbose=True)
    np.testing.assert_almost_equal(np_output[0][1], 0.01189869549125433, decimal=4, verbose=True)
    np.testing.assert_almost_equal(np_output[0][2], 0.044852934777736664, decimal=4, verbose=True)
    np.testing.assert_almost_equal(np_output[0][100], 0.2132769227027893, decimal=4, verbose=True)    
    np.testing.assert_almost_equal(np_output[0][-3], 0.08891487121582031, decimal=4, verbose=True)
    np.testing.assert_almost_equal(np_output[0][-2], 0.006080144084990025, decimal=4, verbose=True)
    np.testing.assert_almost_equal(np_output[0][-1], 0.6892564296722412, decimal=4, verbose=True)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str, default='', help='model name'
    )
    parser.add_argument(
        '--weights_path', type=str, default='', help='path to weights file'
    )
        
    args = parser.parse_args()    
    # use_gpu = torch.cuda.is_available()
    use_gpu = False
        
    print('Building model: {}'.format(args.model))
    model = torchreid.models.build_model(
        name=args.model,
        num_classes=751,        
        pretrained=True,
        use_gpu=use_gpu
    )
    
    load_pretrained_weights(model, args.weights_path)

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    # Load input image
    dir = osp.dirname(osp.realpath(__file__))
    img = torch.load(osp.join(dir, 'testdata/img_tensor.pt'))
    if use_gpu:
        img = img.cuda()

    model.eval()
    with torch.no_grad():
        features = model(img)

    # Test
    test_result(features.cpu().numpy())
    # np.testing.assert_almost_equal(features[0][0], 0, decimal = 4, verbose=True)
    # np.testing.assert_almost_equal(features[0][1], 0.01189869549125433, decimal=4, verbose=True)
    # np.testing.assert_almost_equal(features[0][2], 0.044852934777736664, decimal=4, verbose=True)
    # np.testing.assert_almost_equal(features[0][100], 0.2132769227027893, decimal=4, verbose=True)    
    # np.testing.assert_almost_equal(features[0][-3], 0.08891487121582031, decimal=4, verbose=True)
    # np.testing.assert_almost_equal(features[0][-2], 0.006080144084990025, decimal=4, verbose=True)
    # np.testing.assert_almost_equal(features[0][-1], 0.6892564296722412, decimal=4, verbose=True)

    import onnx
    out_path = '/tmp/resnet50_output.onnx'
    torch.onnx._export(model, img, out_path, input_names=['input0'], output_names=['output0'])
    print('===> Loading and checking exported model')
    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)        
    print("===> Passed")
    
    load_and_infer(img.cpu().numpy())

if __name__ == '__main__':
    main()