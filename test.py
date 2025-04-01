from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets 
import json

# model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
# print(model)

# ds = datasets.load_dataset("cais/wmdp", name="wmdp-bio")
# print(ds)

# # ds = datasets.load_dataset("./wmdp_rephrased/data_english_filler_text", name="default")
# ds = datasets.load_dataset("/root/spar_diogo_test/wmdp_rephrased/data_english_filler_text", name="default")
# print(ds)


with open("wmdp_rephrased/data_translated_vietnamese/test/bio_questions.json", "r") as f:
  lines = f.readlines()
  for i, line in enumerate(lines):
    line = json.loads(line)
    question = line["question"]
    if (question is None) or (question == "null"):
      print(line)
      del lines[i]

with open("wmdp_rephrased/data_translated_vietnamese/test/bio_questions.json", "w") as f:
  for line in lines:
    f.write(line)

