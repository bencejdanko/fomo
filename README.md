# FOMO: Fast Object Localization

FOMO is a lightweight point localization model designed for edge AI applications. Instead of regressing bounding boxes, FOMO downsamples the input image (for example, mapping a 192x192 input to a 24x24 grid) and predicts class probabilities and coordinates on a per-cell basis.

## Installation

Install the package via PyPI:

```bash
pip install fomo-edge-ai
```

## Usage

```python
# Model 
from fomo import FOMO
model = FOMO(model_path=None, size="s", nb_classes=1, device="cpu")

# Training

results = model.train(
    allow_experimental=True,
    data=str(data_yaml_path),   # YOLO style data.yaml
                                # Only bounding-box style datasets supported.
                                # TODO: add support for point-level annotation datasets
    epochs=EPOCHS,
    batch=BATCH,
    lr0=3e-4,
    eval_interval=1,
    workers=2,
    device=device,
    project=PROJECT,
    name=RUN_NAME,
    exist_ok=True,
    patience=0,
)

# Export as TFLite model

fp32_path = trained.export(output_path=str(weights_dir / f"{RUN_NAME}_fp32.tflite"))

# INT8 Quantization

int8_path = quantizer.quantize(
    fp32_tflite=fp32_path,
    calibration_data=calib_iter,
    config=config,
    output_path=str(weights_dir / f"{RUN_NAME}_int8.tflite")
)


```

## Model Hosting

Models are currently available on Hugging Face: 

https://huggingface.co/fomo-edge-ai/FOMO

## Examples

Refer to `examples/` for detailed examples on training and inference.

## Tests 

Tests are completed using [modal](https://modal.com/). Install the modal cli and run

```
make test
```

## License

Code is licensed under the Apache License 2.0. Pre-trained weights are hosted externally and may inherit separate licensing terms. Check details in the specific weight repositories.
