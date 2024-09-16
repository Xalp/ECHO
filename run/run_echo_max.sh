dataset_name="coin_flip"
model_name="gpt-3.5-turbo-0301"
num_clusters=32
iter_CAT=3


# step 3: generate the CAT demo
python run_ECHO_max.py \
--dataset ${dataset_name} \
--method auto_cot \
--demo_path demos/${dataset_name}_${model_name} \
--output_dir ECHO_demos/${dataset_name}_${model_name}_max \
--model ${model_name}

sleep 2

# # step 4: inference and evaluate
# python run_inference.py \
# --dataset $dataset_name \
# --demo_path ECHO_demos/${dataset_name}_${model_name}_max \
# --output_dir experiment/${dataset_name}_${model_name}_${num_clusters}_max \
# --method auto_cot \
# --model $model_name 