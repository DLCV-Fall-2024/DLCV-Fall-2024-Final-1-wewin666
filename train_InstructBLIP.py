from transformers import InstructBlipProcessor, InstructBlipVisionModel, AutoImageProcessor

import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, Trainer, TrainingArguments, DataCollatorForSeq2Seq, BitsAndBytesConfig
from datasets import load_dataset, concatenate_datasets
from PIL import Image
from peft import LoraConfig, get_peft_model, TaskType

from accelerate import dispatch_model, infer_auto_device_map
from transformers.utils import logging



print("torch version:", torch.__version__)
# 1. Load Dataset
# Replace with your Hugging Face dataset name
dataset_name = "ntudlcv/dlcv_2024_final1"
dataset = load_dataset(dataset_name, split="train")


# If want use val also as train data
'''
train_dataset = load_dataset(dataset_name, split="train")
val_dataset = load_dataset(dataset_name, split="val")

# Combine train and val datasets into a single training dataset
dataset = concatenate_datasets([train_dataset, val_dataset])
'''

#dataset = dataset.train_test_split(test_size=0.99)["train"]
# 2. Preprocess the Dataset

processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor.tokenizer.padding_side = "right"


#instruct_blip_processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
instruct_blip_image_processor = AutoImageProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

instruct_blip_image_processor.size = {"height": 336, "width": 336}

instruct_blip_image_processor.patch_size = 14  # Common default for LLaVA models
instruct_blip_image_processor.vision_feature_select_strategy = "default"

class DebugTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Debug input shapes
        print("Debug - Input Shapes:")
        print("Pixel Values Shape:", inputs["pixel_values"].shape)
        print("Input IDs Shape:", inputs["input_ids"].shape)
        print("Attention Mask Shape:", inputs["attention_mask"].shape)
        print("Labels Shape:", inputs["labels"].shape)

        # Move inputs to correct device
        inputs["pixel_values"] = inputs["pixel_values"].to(model.device, dtype=torch.bfloat16)
        inputs["input_ids"] = inputs["input_ids"].to(model.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(model.device)
        inputs["labels"] = inputs["labels"].to(model.device)

        # Step 1: Vision Tower Forward Pass
        with torch.no_grad():
            vision_outputs = model.vision_tower(inputs["pixel_values"])
            vision_hidden_state = vision_outputs.last_hidden_state
            print("Debug - Vision Tower Output Shape:", vision_hidden_state.shape)

        # Step 2: Forward Pass
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                labels=inputs["labels"],
            )

        # Step 3: Check Loss
        if torch.isnan(outputs.loss):
            print("NaN Loss Detected! Dumping Input Data:")
            print("Input IDs:", inputs["input_ids"])
            print("Attention Mask:", inputs["attention_mask"])
            print("Labels:", inputs["labels"])
            print("Pixel Values:", inputs["pixel_values"].shape)
            raise ValueError("Loss became NaN during forward pass.")

        return (outputs.loss, outputs) if return_outputs else outputs.loss
    
def preprocess_function(examples):
    """
    Preprocess function to tokenize text and encode images.
    """
    # User prompt (e.g., instructions)

    text_prompt = examples["conversations"][0]["value"]
    # Assistant response (e.g., description, explanation, or driving advice)
    #print("text prompt", text_prompt)
    text_response = examples["conversations"][1]["value"]

    image = examples["image"]
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")  # Ensure RGB format
    image_inputs = instruct_blip_image_processor(image, return_tensors="pt")
    #print("image_inputs shape:", image_inputs["pixel_values"].shape)
    pixel_values = image_inputs["pixel_values"].to(torch.bfloat16)

    combined_text = f"\nUSER: {text_prompt} \nASSISTANT:{text_response} {processor.tokenizer.eos_token}"

    #combined_text = f"{text_prompt} {text_response} {processor.tokenizer.eos_token}"    
    # Process the image and text prompt
    
    text_inputs = processor.tokenizer(
        combined_text,
        #padding="max_length",       # Fixed-length padding
        truncation=True,            # Truncate if longer than max length
        max_length=512,
        return_tensors="pt"
    )

    input_ids = text_inputs["input_ids"].squeeze(0)
    
    labels = input_ids.clone()
    #input_ids[input_ids == processor.tokenizer.pad_token_id] = -100

    prompt_length = len(processor.tokenizer(text_prompt, add_special_tokens=False)["input_ids"]) + 1  # +1 for SEP token
    labels[:prompt_length] = -100  # Ignore prompt tokens
    #print("Labels when training", labels)

     # Combine the inputs manually
    inputs = {
        "input_ids": input_ids,
        "attention_mask": text_inputs["attention_mask"].squeeze(0),
        "pixel_values": pixel_values.squeeze(0),
        "labels": labels,
    }
    return inputs
   
    

# Apply the preprocessing to the dataset
processed_dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names, num_proc=8)


# 4. Training Arguments
training_args = TrainingArguments(
    output_dir="./llava_finetuned_file_InstructBLIP",     # Directory to save the model
    per_device_train_batch_size=2,      # Adjust batch size based on GPU memory
    gradient_accumulation_steps=4,      # Simulates larger batch size
    learning_rate=2e-4,                 # Standard learning rate for fine-tuning
    num_train_epochs=2,                 # Number of fine-tuning epochs
    logging_dir="./logs",               # Logs directory
    logging_steps=10,                   # Log every 10 steps
    save_strategy="epoch",              # Save model at the end of each epoch
    report_to="none",                   # Disable reporting to external tools
    fp16=False,              # Mixed precision for faster training
    #bf16=True, 
    dataloader_num_workers=4,             # Parallelize data loading
    logging_first_step=True,            # Log the first step
    logging_strategy="steps", 
)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    quantization_config=bnb_config, 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True,
    device_map="auto",
    
)


def force_fp16_hook(module, inputs, output):
    """
    A forward hook to ensure outputs are cast to float16.
    """
    if isinstance(output, torch.Tensor):
        return output.to(torch.bfloat16)
    elif isinstance(output, tuple):
        return tuple(o.to(torch.bfloat16) if isinstance(o, torch.Tensor) else o for o in output)
    else:
        return output

# Register the hook for the vision tower outputs
for submodule in model.vision_tower.modules():
    if isinstance(submodule, torch.nn.Module):
        submodule.register_forward_hook(force_fp16_hook)


lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Causal language model
    r=8,                          # LoRA rank (small rank for efficient fine-tuning)
    lora_alpha=32,                 # Scaling factor for LoRA
    lora_dropout=0.1,              # Dropout to prevent overfitting
    target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]  # Apply LoRA to query/key/value projection layers
)


model = get_peft_model(model, lora_config)


instruct_blip_vision_encoder = InstructBlipVisionModel.from_pretrained("Salesforce/instructblip-vicuna-7b")

#print("Vision tower:", model.vision_tower)
model.vision_tower = instruct_blip_vision_encoder

llava_hidden_size = 1024
instructblip_hidden_size = instruct_blip_vision_encoder.config.hidden_size

#print("llava_hidden_size:", llava_hidden_size)
#print("instructblip_hidden_size:", instructblip_hidden_size)

vision_projector = torch.nn.Linear(instructblip_hidden_size, llava_hidden_size)
model.vision_projector = vision_projector

# Ensure the encoder is frozen
model.vision_tower.eval()
for param in model.vision_tower.parameters():
    param.requires_grad = False


#print("BLIP vision encoder successfully integrated as the LLaVA vision tower.")



# Force LoRA layers to float16
for name, param in model.named_parameters():
    if "lora" in name:  # LoRA layers are named with 'lora'
        param.data = param.data.to(torch.bfloat16)
        param.requires_grad = True



model.print_trainable_parameters()  # Show trainable parameters (should be much fewer now)


# Define the data collator for dynamic padding
data_collator = DataCollatorForSeq2Seq(
    tokenizer=processor.tokenizer,
    model=model,
    padding="longest",  # Force padding to a fixed length
    #max_length=256, 
    pad_to_multiple_of=8  # Optional optimization
)



# 5. Initialize Trainer with data_collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    tokenizer=processor.tokenizer,
    data_collator=data_collator
)


# Step 6: Fine-Tuning the Model
print("Starting fine-tuning with LoRA...")

trainer.train()


# Step 7: Save the Fine-Tuned Model
print("Saving the model...")
model.save_pretrained("./llava_finetuned_lora_InstructBLIP")
processor.save_pretrained("./llava_finetuned_lora_InstructBLIP")

print("Fine-tuning complete! Model saved to './llava_finetuned_lora_InstructBLIP'")