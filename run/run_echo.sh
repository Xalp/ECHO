dataset_name="commonsensqa"
model_name="gpt-3.5-turbo-0301"
ECHO_model_name="gpt-3.5-turbo-0301"
num_clusters=8 
iter_ECHO=2

# step 3: generate the ECHO demo
python run_ECHO.py \
--dataset ${dataset_name} \
--method auto_cot \
--demo_path demos/${dataset_name}_${model_name}_${num_clusters} \
--output_dir ECHO_demos/${dataset_name}_${ECHO_model_name}_${num_clusters}_${iter_ECHO} \
--iter ${iter_ECHO} \
--model ${ECHO_model_name}

# sleep 2

# # step 4: inference and evaluate
# python run_inference.py \
# --dataset $dataset_name \
# --demo_path ECHO_demos/${dataset_name}_${ECHO_model_name}_${num_clusters}_${iter_ECHO} \
# --output_dir experiment/${dataset_name}_${model_name}_${num_clusters}_${iter_ECHO} \
# --method auto_cot \
# --model $model_name 