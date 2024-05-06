import torch
from transformers import AutoModelForSequenceClassification, AutoConfig

# Load the pre-quantized Phi2 model from Hugging Face
model_name = "phi2-base-uncased-quantized-4bit"
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token="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")

# Verify that the model is quantized
print("Model quantization dtype:", model.quantization_dtype) # Should print qint8

# Use the quantized model for inference
input_ids = torch.tensor([[1, 2, 3, 4, 5]]) # Example input IDs
attention_mask = torch.tensor([[1, 1, 1, 1, 1]]) # Example attention mask
outputs = model(input_ids, attention_mask=attention_mask)
print("Output shape:", outputs.logits.shape)