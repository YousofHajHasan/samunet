CUDA_VISIBLE_DEVICES="0" \
python train.py \
--hiera_path "sam2_hiera_large.pt" \
--train_image_path "sam_data_multiclass/train/images/" \
--train_mask_path "sam_data_multiclass/train/masks/" \
--val_image_path "sam_data_multiclass/val/images/" \
--val_mask_path "sam_data_multiclass/val/masks/" \
--save_path "checkpoints/" \
--num_classes 7 \
--input_size 512 \
--epoch 20 \
--batch_size 6 \
--lr 0.001
  


