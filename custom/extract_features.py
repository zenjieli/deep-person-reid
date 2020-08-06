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

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str, default='', help='model name'
    )
    parser.add_argument(
        '--weights_path', type=str, default='', help='path to weights file'
    )
        
    args = parser.parse_args()    
    use_gpu = torch.cuda.is_available()
        
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
    np.testing.assert_almost_equal(features[0][0].item(), 0, decimal = 4, verbose=True)
    np.testing.assert_almost_equal(features[0][1].item(), 0.01189869549125433, decimal=4, verbose=True)
    np.testing.assert_almost_equal(features[0][2].item(), 0.044852934777736664, decimal=4, verbose=True)
    np.testing.assert_almost_equal(features[0][100].item(), 0.2132769227027893, decimal=4, verbose=True)    
    np.testing.assert_almost_equal(features[0][-3].item(), 0.08891487121582031, decimal=4, verbose=True)
    np.testing.assert_almost_equal(features[0][-2].item(), 0.006080144084990025, decimal=4, verbose=True)
    np.testing.assert_almost_equal(features[0][-1].item(), 0.6892564296722412, decimal=4, verbose=True)
    
    print("Passed")

if __name__ == '__main__':
    main()