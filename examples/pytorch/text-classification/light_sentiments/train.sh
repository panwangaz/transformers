CUDA_VISIBLE_DEVICES=1 python examples/pytorch/text-classification/light_sentiments/run_glue_new.py \
	--my_train_dir /data/yangxiaoran/nonrt_init/pay_rate_level_train/data/sentiment_train_new/ \
	--my_validataion_dir /data/yangxiaoran/nonrt_init/pay_rate_level_train/data/sentiment_valid_new/ \
	--model_name_or_path ckpts/bert-base-uncased \
	--task_name sst2 \
	--do_train True \
	--do_eval True \
	--do_predict True \
	--per_device_train_batch_size 16 \
	--per_device_eval_batch_size 16 \
	--learning_rate 1e-5 \
	--num_train_epochs 5 \
	--logging_strategy steps \
	--logging_first_step True \
	--logging_steps 200 \
	--evaluation_strategy steps \
	--eval_steps 800 \
	--overwrite_output_dir True \
	--output_dir ./tmp/sst2/ \
	--logging_dir ./tmp/sst2/model_diff/bert_only_user_msc/ \
	--ignore_mismatched_sizes True \
	--fp16 True \
	--fp16_opt_level O1 \
	--lr_scheduler_type cosine \
	--warmup_ratio 0.20 \
	--save_strategy epoch \
	--save_steps 1 \
	--save_model_path ./tmp/detect_flirt.finbert.v2.model/ \
	--balance True \
	--regression False \
	--use_special_token True \
	--kick_ratio 0
