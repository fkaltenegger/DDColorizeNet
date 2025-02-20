# ColorizeNet

This model is a ControlNet based on SD-v2.1, trained for image colorization from black and white images.
## Model Details

### Model Description

ColorizeNet is an image colorization model based on ControlNet, trained using the pre-trained Stable Diffusion model version 2.1 proposed by Stability AI.

- **Finetuned from model :** [https://huggingface.co/stabilityai/stable-diffusion-2-1]

## Usage

### Training Data

The model has been trained on [COCO](https://huggingface.co/datasets/detection-datasets/coco), using all the images in the dataset and converting them to grayscale to use them to condition the ControlNet. To train the model, you also need a JSON file specifying the input prompt and the source and target images. The file used for training is reported in `data/colorization/training/train.json`. Prompts were obtained by randomly choosing one among similar prompts for each image pair generated with [Instruct BLIP](https://huggingface.co/docs/transformers/main/en/model_doc/instructblip).

You can download the datasets for training by running the script `download_data.sh`

```bash
bash download_data.sh
```

### Inference

### Training

#### Download the original Stable Diffusion-v2.1 weights

Download the weights of the original SD models to init ControlNet

```bash
bash download_models.sh
```
The weights will be placed under the `models` folder

#### Create the weights of the ControlNet

Launch the script to create the controlnet weights by specifying the original weights and the path where to save the modified ones

```bash
python tool_add_control.py weights/sd/v2-1_768-ema-pruned.pt weights/controlnet/v2-1_colorization.pt 
```

#### Train the model

```bash
python train.py
```

## Additional Resources
For more information on ControlNet, please refer to the [original repository](https://github.com/lllyasviel/ControlNet)
