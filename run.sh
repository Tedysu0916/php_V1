##Test
CUDA_VISIBLE_DEVICES=0 \
python test.py \
--config_file './your path/logs/CUHK-PEDES/sdm+itc+aux_cnum9/configs.yaml'


##Trainging
#CUDA_VISIBLE_DEVICES=5 \
#python train.py \
#--name php \
#--img_aug \
#--batch_size 64 \
#--loss_names 'sdm+itc+aux' \
#--dataset_name 'CUHK-PEDES' \
#--root_dir "./your path/data" \
#--num_epoch 60 \
#--num_experts 4 \
#--topk 2 \
#--reduction 8 \
#--moe_layers 4 \
#--moe_heads 8 \
#--transformer_lr_factor 1.0 \
#--moe_lr_factor 2.0 \
#--aux_factor 3.0 \
#--lr 3e-6 \
#--cnum 9



#ICFG-PEDES CUHK-PEDES RSTPReid


