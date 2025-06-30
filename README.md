# IBench: Image Generation Evaluation Framework

IBench is a comprehensive framework for evaluating image generation quality and image-text alignment. It provides multiple metrics across different evaluation dimensions.

## Features

### Text-to-Image (T2I) Metrics
- **FID Score**: Evaluates the fidelity and diversity of generated images.
- **Aesthetic Score**: Measures image aesthetic quality using the LAION aesthetic model.
- **Imaging Quality**: Analyzes technical image quality using the MUSIQ model.

### Image-ID Evaluation
- **Face Similarity**: Measures facial similarity using InsightFace.
- **CLIP Evaluation**: Assesses semantic similarity between images using CLIP.
- **DINO Evaluation**: Analyzes visual features using DINOv2.
- **DreamSim Evaluation**: Assesses dream-like quality.
- **Pose Diversity**: Analyzes head pose variations using Hopenet.
- **Expression Diversity**: Evaluates facial expression variations.

### Multi-modal LLM Evaluation
- **GPT-4V Integration**: Scores image-text alignment.
- Configurable evaluation parameters.
- JSON-formatted analysis output.

## Configuration

The framework uses YAML configuration for model paths and parameters:

```yaml
metrics:
  t2i:
    fid:
      fid_inception_model_path: "/path/to/inception/model"
    aesthetic:
      laion_aes_model_path: "/path/to/laion/model"
  imageid:
    facesim:
      face_detection_model_path: "/path/to/insightface"
    clipeval:
      clip_model_path: "/path/to/clip"
    dinoeval:
      dino_model_path: "/path/to/dino"
    dreamsimeval:
      dreamsim_model_path: "/path/to/dreamsim"
    posediv:
      hopenet_model_path: "/path/to/hopenet"
      yaw_threshold: 45
      pitch_threshold: 20
      roll_threshold: 25
    exprdiv:
      expression_model_path: "/path/to/expression"
  mllm:
    gpt_proxy_url: "http://proxy.url:port"
    temperature: 0.02
    max_tokens: 250
    top_p: 1
    frequency_penalty: 0
    presence_penalty: 0

enable_timing_stats: True
save_results: "/path/to/results"
```

## Usage

1. Install dependencies.
2. Configure model paths in `config/config.yaml`.
3. Run evaluation:

```python
from ibench.metrics.mllm.gpt import GPT

evaluator = GPT(model="gpt-4v", user_prompt="")
score = evaluator.evaluate(data)
```

## Additional Features

- Modular architecture for easy metric addition.
- YAML-based configuration.
- Support for batch processing.
- Detailed scoring and analysis.
- Progress tracking with `tqdm`.

## Requirements

- Python 3.7+
- OpenAI API key for GPT-4V evaluation.
- Pre-trained models (CLIP, DINOv2, etc.).
