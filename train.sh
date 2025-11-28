CUDA_VISIBLE_DEVICES="0" \
python train.py \
--hiera_path "sam2_hiera_large.pt" \
--train_image_path "/home/yousof/Fine-Tuning-SAM2-UNet-for-Breast-Image-Segmentation-Without-Pectoral-Muscle-and-Artifacts/sam_data/train/images/" \
--train_mask_path "/home/yousof/Fine-Tuning-SAM2-UNet-for-Breast-Image-Segmentation-Without-Pectoral-Muscle-and-Artifacts/sam_data/train/masks/" \
--val_image_path "/home/yousof/Fine-Tuning-SAM2-UNet-for-Breast-Image-Segmentation-Without-Pectoral-Muscle-and-Artifacts/sam_data/val/images/" \
--val_mask_path "/home/yousof/Fine-Tuning-SAM2-UNet-for-Breast-Image-Segmentation-Without-Pectoral-Muscle-and-Artifacts/sam_data/val/masks/" \
--save_path "checkpoint" \
--epoch 50 \
--lr 0.005 \
--batch_size 4