CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "/home/yousof/Downloads/35epochs_0008lr_28batchsize_512inputsize.pth" \
--test_image_path "/home/yousof/Desktop/samunet/samunet/new_data_coco_multiclass/train/images/" \
--test_gt_path "/home/yousof/Desktop/samunet/samunet/new_data_coco_multiclass/train/masks/" \
--save_path "/home/yousof/Desktop/samunet/samunet/new_data_coco_multiclass/train/predictions/" \
--num_classes 7 \
--input_size 512