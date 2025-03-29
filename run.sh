set -e 

# below creates a log file
# lm_eval \
#   --model hf \
#   --model_args pretrained=HuggingFaceH4/zephyr-7b-beta \
#   --tasks wmdp \
#   --limit 10 \
#   --output output/wmdp/ \
#   --log_samples

# python3 -m eval.eval --model zephyr_7b_beta --tasks wmdp_bio wmdp_cyber wmdp_chem --output output/wmdp/ --log_samples
# python3 -m eval.eval --model zephyr_7b_beta --tasks tinyMMLU --output output/tinyMMLU/ --log_samples


python3 -m eval.eval --model zephyr_7b_beta --tasks wmdp_rephrased --output output/wmdp_rephrased/ --log_samples