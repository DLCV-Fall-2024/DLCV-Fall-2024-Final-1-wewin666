import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoImageProcessor, InstructBlipVisionModel
from datasets import load_dataset
from PIL import Image
import json
from peft import PeftModel
from torch.utils.data import DataLoader
import sys

# 1. Load Fine-Tuned Model and Processor
model_path = "./checkpoint-7202"  # Path to the fine-tuned model
model = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")


'''
llava_hidden_size = 1024
instructblip_hidden_size = instruct_blip_vision_encoder.config.hidden_size

print("llava_hidden_size:", llava_hidden_size)
print("instructblip_hidden_size:", instructblip_hidden_size)

vision_projector = torch.nn.Linear(instructblip_hidden_size, llava_hidden_size)
model.vision_projector = vision_projector
'''

# Ensure the encoder is frozen
model.vision_tower.eval()
for param in model.vision_tower.parameters():
    param.requires_grad = False


processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

instruct_blip_image_processor = AutoImageProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

instruct_blip_image_processor.size = {"height": 336, "width": 336}

instruct_blip_image_processor.patch_size = 14  # Common default for LLaVA models
instruct_blip_image_processor.vision_feature_select_strategy = "default"

#print("Expected resolution:", instruct_blip_image_processor.size)

# 2. Load Validation Dataset (1% of Data for Testing)

from datasets import load_from_disk

# Specify the path where the dataset is saved in Google Drive
dataset_name = "ntudlcv/dlcv_2024_final1"
dataset = load_dataset(dataset_name, split="test")
#dataset = dataset.train_test_split(test_size=0.95)["train"]  # 1% subset for quick testing

# Custom collate function
def custom_collate_fn(batch):
    """
    Custom collate function to handle PIL.Image objects.
    Converts images into tensors and retains other fields as is.
    """
    # Extract and preprocess data for batching
    batch_images = []
    batch_conversations = []
    batch_ids = []

    for example in batch:
        # Convert image to tensor
        image = example["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        batch_images.append(image)

        # Collect other fields
        batch_conversations.append(example["conversations"])
        batch_ids.append(example["id"])

    return {
        "images": batch_images,
        "conversations": batch_conversations,
        "ids": batch_ids,
    }
# Create DataLoader with custom collate_fn
batch_size = 10
data_loader = DataLoader(dataset, batch_size=batch_size , shuffle=False, collate_fn=custom_collate_fn)


def clean_assistant_response(response):
    """
    Extracts the assistant's response by removing the 'ASSISTANT:' prefix and any leading/trailing spaces.
    """
    # Remove everything up to and including "ASSISTANT:"
    if "ASSISTANT:" in response:
        response = response.split("ASSISTANT:", 1)[-1].strip()
    return response
# 3. Define Inference Function
def infer_batch(batch):
    """
    Runs inference on a batch of examples.
    """
    # Prepare text prompts
    text_prompts = [
        f"\nUSER: {example[0]['value']} \nASSISTANT:"
        for example in batch['conversations']
    ]
    '''
    for example in batch['conversations']:
      print("prompt:", example[0]['value'])
    '''

    # Process images
    images = [
        Image.open(image).convert("RGB")
        if not isinstance(image, Image.Image) else image
        for image in batch['images']
    ]
    image_inputs = instruct_blip_image_processor(images, return_tensors="pt", padding=True)

    # Process text
    text_inputs = processor.tokenizer(
        text_prompts,
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors="pt"
    )

    # Move inputs to the model's device
    inputs = {
        "input_ids": text_inputs["input_ids"].to(model.device),
        "attention_mask": text_inputs["attention_mask"].to(model.device),
        "pixel_values": image_inputs["pixel_values"].to(model.device),
    }

    # Generate the model's response
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            max_length=1024,
            do_sample=True,
            temperature=0.001,
            repetition_penalty=1.2,
        )

    # Decode the responses
    responses = [
        clean_assistant_response(processor.tokenizer.decode(output, skip_special_tokens=True))
        for output in output_ids
    ]
    return responses


output_file = "submission_InstructBLIP.json"
output_file = sys.argv[1]
#print("output_file:", output_file)

output_dict = {}
# Open the JSON file in append mode
with open(output_file, "w") as f:
    f.write("{\n")  # Start the JSON file as an open dictionary
first_entry = True  # Track whether this is the first entry for proper JSON formatting
#print("\nRunning batch inference on the test dataset...\n")
#print("=" * 80)
for batch_idx, batch in enumerate(data_loader):
    # Access data from the custom collate_fn output
    example_ids = batch["ids"]
    user_prompts = [conv[0]["value"] for conv in batch["conversations"]]

    # Perform inference
    predicted_responses = infer_batch(batch)

    # Save incrementally to the file
    with open(output_file, "a") as f:
        for i, example_id in enumerate(example_ids):
            if not first_entry:
                f.write(",\n")
            f.write(f'    "{example_id}": {json.dumps(predicted_responses[i])}')
            first_entry = False
    '''
    # Display results
    for i, user_prompt in enumerate(user_prompts):
        print(f"Batch {batch_idx+1}, Example {i+1}:")
        print("-" * 80)
        print(f"User Prompt:\n{user_prompt}\n")
        print(f"Model Response:\n{predicted_responses[i]}\n")
        print("-" * 80)
    '''
with open(output_file, "a") as f:
    f.write("\n}\n")

#print("\nBatch inference complete!")
#print("=" * 80)


