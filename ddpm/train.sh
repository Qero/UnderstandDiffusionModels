python3 train.py \
--max_sample_step=1000 \
--data_dir="./dataset" \
--log_dir="./logs" \
--model_dir="./checkpoints" \
--num_gpu=4 \
--num_loaders=50 \
--num_epoch=100 \
--batch_size=128 \
--img_size=128
