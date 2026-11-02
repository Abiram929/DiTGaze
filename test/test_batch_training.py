import torch
import sys
import argparse
from configs.gazegaussian_options import BaseOptions
from dataloader.eth_xgaze import get_train_loader
from trainer.gazegaussian_trainer import GazeGaussianTrainer
from utils.recorder import GazeGaussianTrainRecorder

def test_batch_sizes():
    base_options = BaseOptions()
    
    print("=" * 80)
    print("TESTING BATCH SIZE COMPATIBILITY")
    print("=" * 80)
    
    for batch_size in [1, 2, 4]:
        print(f"\n{'='*60}")
        print(f"Testing with batch_size={batch_size}")
        print(f"{'='*60}")
        
        try:
            base_options.batch_size = batch_size
            base_options.num_workers = 0
            base_options.num_epochs = 1
            
            train_data_loader = get_train_loader(
                base_options, 
                data_dir=base_options.img_dir, 
                batch_size=batch_size, 
                num_workers=0, 
                evaluate="landmark", 
                is_shuffle=False,
                dataset_name=base_options.dataset_name
            )
            
            recorder = GazeGaussianTrainRecorder(base_options)
            trainer = GazeGaussianTrainer(base_options, recorder)
            
            for batch_idx, data in enumerate(train_data_loader):
                if batch_idx >= 2:
                    break
                    
                print(f"\n  Batch {batch_idx + 1}:")
                print(f"    Image shape: {data['image'].shape}")
                print(f"    Batch size from data: {data['image'].shape[0]}")
                
                data = trainer.prepare_data(data)
                data = trainer.build_code_and_cam(data["idx_input"][0], data)
                
                with torch.set_grad_enabled(False):
                    data = trainer.net(data)
                    
                print(f"    ✓ Forward pass successful")
                print(f"    Output shapes:")
                print(f"      - merge_img_pro: {data['total_render_dict']['merge_img_pro'].shape}")
                print(f"      - merge_img: {data['total_render_dict']['merge_img'].shape}")
                
                batch_loss_dict = trainer.loss_utils.calc_total_loss(
                    data,
                    epoch=0,
                    batch_num=batch_idx,
                    discriminator=trainer.discriminator,
                )
                
                print(f"    ✓ Loss computation successful")
                print(f"    Total loss: {batch_loss_dict['total_loss'].item():.4f}")
                
            print(f"\n  ✓✓✓ Batch size {batch_size} PASSED ✓✓✓")
            
        except Exception as e:
            print(f"\n  ✗✗✗ Batch size {batch_size} FAILED ✗✗✗")
            print(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()
            
    print(f"\n{'='*80}")
    print("BATCH SIZE TESTING COMPLETE")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    test_batch_sizes()
