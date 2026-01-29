#!/usr/bin/env python3
"""
Simple denoiser wrapper script that takes input/output paths as arguments
Usage: python denoise_wrapper.py <input_image_path> <output_image_path>
"""

import os
import sys
import shutil
from pathlib import Path

# Get the directory of this file and resolve paths relative to it
wrapper_dir = Path(__file__).parent  # backend/
denoiser_ffdnet_dir = wrapper_dir / 'denoiser' / 'ffdnet'

# Add denoiser path
sys.path.insert(0, str(denoiser_ffdnet_dir))

def denoise_single_image(input_path: str, output_path: str):
    """Denoise a single image using FFDNet"""
    
    os.chdir(str(denoiser_ffdnet_dir))
    
    # Create temp testsets folder
    temp_testsets = 'testsets/webapp_denoise'
    os.makedirs(temp_testsets, exist_ok=True)
    
    # Copy input image
    image_name = os.path.basename(input_path)
    temp_input = os.path.join(temp_testsets, image_name)
    shutil.copy(input_path, temp_input)
    
    # Now run the main denoiser function with hardcoded testset name
    import logging
    import numpy as np
    from collections import OrderedDict
    import torch
    from utils import utils_logger, utils_image as util
    
    # ========== Denoiser parameters ==========
    noise_level_img = 5
    noise_level_model = noise_level_img
    model_name = 'ffdnet_color'
    testset_name = 'webapp_denoise'
    need_degradation = False
    show_img = False
    
    task_current = 'dn'
    sf = 1
    n_channels = 3
    nc = 96
    nb = 12
    use_clip = False
    model_pool = 'model_zoo'
    testsets = 'testsets'
    results = 'results'
    result_name = testset_name + '_' + model_name
    border = 0
    model_path = os.path.join(model_pool, model_name+'.pth')
    
    # ========== Setup paths ==========
    L_path = os.path.join(testsets, testset_name)
    H_path = L_path
    E_path = os.path.join(results, result_name)
    util.mkdir(E_path)
    
    need_H = True if H_path is not None else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ========== Load model ==========
    from models.network_ffdnet import FFDNet as net
    model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    
    # ========== Process images ==========
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    
    L_paths = util.get_image_paths(L_path)
    
    for idx, img in enumerate(L_paths):
        img_name, ext = os.path.splitext(os.path.basename(img))
        img_L = util.imread_uint(img, n_channels=n_channels)
        img_L = util.uint2single(img_L)
        
        if need_degradation:
            np.random.seed(seed=0)
            img_L += np.random.normal(0, noise_level_img/255., img_L.shape)
            if use_clip:
                img_L = util.uint2single(util.single2uint(img_L))
        
        img_L = util.single2tensor4(img_L)
        img_L = img_L.to(device)
        sigma = torch.full((1,1,1,1), noise_level_model/255.).type_as(img_L)
        
        img_E = model(img_L, sigma)
        img_E = util.tensor2uint(img_E)
        
        # Save denoised image
        util.imsave(img_E, os.path.join(E_path, img_name+ext))
    
    # ========== Copy output to final location ==========
    denoised_temp = os.path.join(E_path, image_name)
    shutil.copy(denoised_temp, output_path)
    
    # Cleanup
    shutil.rmtree(temp_testsets, ignore_errors=True)
    shutil.rmtree(E_path, ignore_errors=True)
    
    print(f"SUCCESS: Denoised image saved to {output_path}")
    return output_path

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python denoise_wrapper.py <input_path> <output_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    try:
        denoise_single_image(input_path, output_path)
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
