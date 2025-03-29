from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
print(model)
