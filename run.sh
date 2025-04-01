set -e 

if [ ! -d "lm-evaluation-harness" ]; then
  git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
  cd lm-evaluation-harness
  uv pip install -e .
  cd ..
fi


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

export LOGLEVEL=DEBUG

# lm_eval --model hf --model_args pretrained=HuggingFaceH4/zephyr-7b-beta --tasks wmdp_bio_rephrased --log_samples --output output/wmdp_rephrased
# lm_eval --model hf --model_args pretrained=cais/zephyr_rmu --tasks wmdp_bio_rephrased --log_samples --output output/wmdp_rephrased

# python3 -m eval.eval --model zephyr_7b_rmu --tasks wmdp_bio_rephrased_english_filler --output output/wmdp_rephrased --log_samples
# python3 -m eval.eval --model zephyr_7b_rmu --tasks wmdp_bio_rephrased_hindi_filler --output output/wmdp_rephrased --log_samples
# python3 -m eval.eval --model zephyr_7b_rmu --tasks wmdp_bio_rephrased_latin_filler --output output/wmdp_rephrased --log_samples
# python3 -m eval.eval --model zephyr_7b_rmu --tasks wmdp_bio_rephrased_conversation --output output/wmdp_rephrased --log_samples
# python3 -m eval.eval --model zephyr_7b_rmu --tasks wmdp_bio_rephrased_poem --output output/wmdp_rephrased --log_samples
# python3 -m eval.eval --model zephyr_7b_rmu --tasks wmdp_bio_rephrased_replace_with_variables --output output/wmdp_rephrased --log_samples
# python3 -m eval.eval --model zephyr_7b_rmu --tasks wmdp_bio_rephrased_technical_terms_removed_1 --output output/wmdp_rephrased --log_samples
# python3 -m eval.eval --model zephyr_7b_rmu --tasks wmdp_bio_rephrased_translated_arabic --output output/wmdp_rephrased --log_samples
# python3 -m eval.eval --model zephyr_7b_rmu --tasks wmdp_bio_rephrased_translated_bengali --output output/wmdp_rephrased --log_samples
# python3 -m eval.eval --model zephyr_7b_rmu --tasks wmdp_bio_rephrased_translated_czech --output output/wmdp_rephrased --log_samples
# python3 -m eval.eval --model zephyr_7b_rmu --tasks wmdp_bio_rephrased_translated_farsi --output output/wmdp_rephrased --log_samples
# python3 -m eval.eval --model zephyr_7b_rmu --tasks wmdp_bio_rephrased_translated_french --output output/wmdp_rephrased --log_samples
# python3 -m eval.eval --model zephyr_7b_rmu --tasks wmdp_bio_rephrased_translated_german --output output/wmdp_rephrased --log_samples
# python3 -m eval.eval --model zephyr_7b_rmu --tasks wmdp_bio_rephrased_translated_hindi --output output/wmdp_rephrased --log_samples
# python3 -m eval.eval --model zephyr_7b_rmu --tasks wmdp_bio_rephrased_translated_korean --output output/wmdp_rephrased --log_samples
# python3 -m eval.eval --model zephyr_7b_rmu --tasks wmdp_bio_rephrased_translated_telugu --output output/wmdp_rephrased --log_samples
# python3 -m eval.eval --model zephyr_7b_rmu --tasks wmdp_bio_rephrased_translated_turkish --output output/wmdp_rephrased --log_samples
# python3 -m eval.eval --model zephyr_7b_rmu --tasks wmdp_bio_rephrased_translated_vietnamese --output output/wmdp_rephrased --log_samples

# python3 -m eval.eval \
#   --model zephyr_7b_beta \
#   --tasks wmdp_bio wmdp_bio_rephrased_english_filler wmdp_bio_rephrased_hindi_filler wmdp_bio_rephrased_latin_filler wmdp_bio_rephrased_conversation wmdp_bio_rephrased_poem wmdp_bio_rephrased_replace_with_variables wmdp_bio_rephrased_technical_terms_removed_1 wmdp_bio_rephrased_translated_arabic wmdp_bio_rephrased_translated_bengali wmdp_bio_rephrased_translated_czech wmdp_bio_rephrased_translated_farsi wmdp_bio_rephrased_translated_french wmdp_bio_rephrased_translated_german wmdp_bio_rephrased_translated_hindi wmdp_bio_rephrased_translated_korean wmdp_bio_rephrased_translated_telugu wmdp_bio_rephrased_translated_turkish wmdp_bio_rephrased_translated_vietnamese \
#   --output output/wmdp_rephrased \
#   --log_samples