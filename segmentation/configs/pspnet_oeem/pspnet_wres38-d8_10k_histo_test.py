_base_ = './pspnet_wres38-d8_10k_histo.py'

test_cfg = dict(mode='slide', crop_size=(320, 320), stride=(256, 256), crf=False,
                pred_output_path='glas/test_patches', npy=True)
