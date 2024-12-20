# Fine-tuning Stable Diffusion [SDXL - QLORA] on Custom Dataset for Image Generation  🖋️ 🚀 🖼️


## <br>**➲ Project Overview** :

The project focuses on training and optimizing a Stable Diffusion model for specific image generation tasks using transfer learning techniques.

## <br>**➲ Dataset** :

-  [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) for face images.

---
-- base_model: stabilityai/stable-diffusion-xl-base-1.0
-- library_name: diffusers


<strong>Goal of this project:</strong> This project focuses on building an advanced text-to-image generation system using the Stable Diffusion XL (SDXL) model, a state-of-the-art deep learning architecture. The goal is to transform natural language text descriptions into visually coherent and high-quality images, unlocking creative possibilities in areas like art generation, design prototyping, and multimedia applications.

To enhance performance and tailor the model to specific use cases, SDXL is fine-tuned using <strong>QLoRA (Quantized Low-Rank Adaptation)</strong>. This approach leverages efficient parameter fine-tuning and memory optimization techniques, enabling high-quality adaptations with reduced computational overhead. Fine-tuning with QLoRA ensures that the model is optimized for domain-specific text-to-image tasks, delivering even more precise and creative outputs.

## <br>**➲ Simplified Architecture** :

<img src="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/pipeline.png" alt="Generated Image 1" style="max-width: 35%; height: 250px; border: 2px solid #ccc; border-radius: 8px; display: inline-block; margin-right: 10px;">

## <br>**➲ Dataset Description: CelebFaces Attributes Dataset (CelebA)** :

<p style="font-family: Lucida Sans ;font-size:15px;">The CelebA dataset is a widely-used, large-scale dataset in the field of computer vision, particularly for tasks related to faces. It consists of over 200,000 celebrity face images annotated with a rich set of attributes. The dataset offers diverse visual content with variations in pose, facial expressions, and backgrounds, making it suitable for a range of face-related applications.</p>


## <br>**➲ Here are few examples of generated images Using Stable Diffusion SDXL:

<strong>Before Fine-Tuning SDXL</strong><br>

<img src="./generated_img1.png" alt="Generated Image 1" style="max-width: 35%; height: 220px; border: 2px solid #ccc; border-radius: 8px; display: inline-block; margin-right: 10px;"><br><br>
<strong>After Fine-Tuning SDXL on Custom Dataset</strong><br><br>
<img src="./after_training_img1.png" alt="After Fine-Tuning Image 1" style="max-width: 35%; height: 220px; border: 2px solid #ccc; border-radius: 8px; display: inline-block; margin-right: 10px;">
<img src="./after_training_img2.png" alt="After Fine-Tuning Image 2" style="max-width: 35%; height: 220px; border: 2px solid #ccc; border-radius: 8px; display: inline-block;">


## <br>**➲ How to use the fine-tuned model for image generation**:  

### <br>**➲ Loading Pre-trained Model and Fine-Tuned LoRA Weights** :

This section demonstrates how to load the pre-trained Stable Diffusion XL model and the fine-tuned LoRA weights for generating high-quality images based on text prompts.

### Prerequisites
Ensure you have the necessary dependencies installed. You can install them via:
import torch
from diffusers import DiffusionPipeline

### <br>**➲ Path to the directory containing fine-tuned LoRA weights** :
    -- model_path = "Shuhaib73/stablediffusion_fld"

### Load the pre-trained Stable Diffusion XL model

```python
>> import torch
from diffusers import DiffusionPipeline
```

```python
>>> model_path = "Shuhaib73/stablediffusion_fld"
```

```python
>>> trained_pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
```

### Move the pipeline to GPU for faster processing
```python
>>> trained_pipe.to("cuda")
```

## Load the fine-tuned LoRA weights into the pipeline
```python
>>> trained_pipe.load_lora_weights(model_path)
```

### Generate an image 
```python
>>> generated_images = trained_pipe(
    prompt = "A young woman with long, straight hair, wearing elegant earrings. Her calm expression and stylish outfit complement her natural beauty, with a softly blurred background adding a touch of depth."
)
```


## <br>**➲ Dockerizing and Building a Website using Flask:
    --- To make the model accessible to a wider audience, the system is containerized using Docker, which simplifies deployment and ensures that the application runs consistently across different environments. Docker will package the model, its dependencies, and the web server into a container, allowing for easy scalability and deployment.
    --- Additionally, a Flask-based web application is developed to provide a user-friendly interface for interacting with the model. The website will allow users to input text descriptions and receive generated images in real-time. Flask will handle the routing, user interaction, and model inference, while Docker ensures the system is deployed seamlessly, enabling easy access through a web browser.

    
