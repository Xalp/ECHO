model_name="gpt-3.5-turbo-0301"
ECHO_model_name="gpt-3.5-turbo-0301"

# step 3: generate the ECHO demo
python run_ECHO_plus.py \
--dataset ${dataset_name} \
--method auto_cot \
--demo_path demos/${dataset_name}_${model_name} \
--output_dir ECHO_demos/${dataset_name}_${ECHO_model_name}_max \
--model ${ECHO_model_name}

sleep 2

# step 4: inference and evaluate
python run_inference.py \
--dataset $dataset_name \
--demo_path ECHO_demos/${dataset_name}_${ECHO_model_name}_max \
--output_dir experiment/${dataset_name}_${model_name}_max \
--method auto_cot \
--model $model_name 