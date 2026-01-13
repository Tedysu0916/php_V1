##testing
CUDA_VISIBLE_DEVICES=0 \
python test.py \
--config_file 'logs/ICFG-PEDES/php_training_reproduce_on_IRRA/configs.yaml'


##Trainging
#CUDA_VISIBLE_DEVICES=2 \
#python train.py \
#--name php \
#--img_aug \
#--batch_size 64 \
#--loss_names 'sdm+itc+aux' \
#--dataset_name 'ICFG-PEDES' \
#--root_dir "/media/jqzhu/哈斯提·基拉/UniMoESE/data" \
#--num_epoch 60 \
#--num_experts 4 \
#--topk 2 \
#--reduction 8 \
#--moe_layers 4 \
#--moe_heads 8 \
#--transformer_lr_factor 1.0 \
#--moe_lr_factor 2.0 \
#--aux_factor 0.5 \
#--lr 3e-6 \
#--cnum 9
