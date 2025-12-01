CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "/home/vuedale/Desktop/sam2.1/Fine-Tuning-SAM2-UNet-for-Breast-Image-Segmentation-Without-Pectoral-Muscle-and-Artifacts/checkpoints/best_model_epoch-12.pth" \
--test_image_path "/home/vuedale/Desktop/sam2.1/Fine-Tuning-SAM2-UNet-for-Breast-Image-Segmentation-Without-Pectoral-Muscle-and-Artifacts/sam_data_multiclass/test/images/" \
--test_gt_path "/home/vuedale/Desktop/sam2.1/Fine-Tuning-SAM2-UNet-for-Breast-Image-Segmentation-Without-Pectoral-Muscle-and-Artifacts/sam_data_multiclass/test/masks/" \
--save_path "/home/vuedale/Desktop/sam2.1/Fine-Tuning-SAM2-UNet-for-Breast-Image-Segmentation-Without-Pectoral-Muscle-and-Artifacts/sam_data_multiclass/test/predictions/" \
--num_classes 7 \
--input_size 512