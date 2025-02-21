lang=javascript
folder=javascript
pathTrainTestData="../../datasets_models_unixcoder/"
mkdir -p $pathTrainTestData/eval_uni_test/$lang
python finetune_search_gen.py \
    --output_dir $pathTrainTestData/eval_uni_test/$folder/ \
    --model_name_or_path microsoft/unixcoder-base  \
    --output_vector_model=$pathTrainTestData/vectors/ \
    --do_test \
    --test_data_file $pathTrainTestData/datasets/$lang/test.jsonl \
    --codebase_file $pathTrainTestData/datasets/$lang/codebase.jsonl \
    --num_train_epochs 30 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 48 \
    --eval_batch_size 48 \
    --learning_rate 2e-5 \
    --seed 123456
