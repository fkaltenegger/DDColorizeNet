# Differential Diffusion ColorizeNet 

This colorization model is a combination of ColorizeNet, which is a Stable Diffusion Model based on ControlNet, and Differential Diffusion.

## Model Components

The edge maps guiding the colorization process were created by using the learned edge detector CATS (https://github.com/WHUHLX/CATS). If you want to use HED edge maps, an HED edge detector is implemented from ControlNet.

## Environment Setup & Download of Pretrained Model
First create a new environment:

    conda env create -f environment.yaml
    conda activate control

<b>Note that you might encounter some version mismatches.</b>

Afterwards you can download the model by executing:

    download_colorizenet.sh

## Colorization of Images

The grayscale images and the CATS edge maps are located in the following folders:

    "/test_imgs/ddcolorizenet"
    "/test_imgs/cats"

To colorize images with DDColorizeNet, execute the following script with the specified parameters:

    python3 ddcolorize.py <image_name>.<file_extension>

### HED-based Edge Maps
If you want to use HED-based edge maps, just remove the parameters <b>input_mask_path</b> and <b>input_mask</b> from the <b>colorize_image</b> function call.

### Dynamic Masking - Differential Diffusion
Dynamic Masking is the default setting because it performs better than the threshold-based approach.

### Threshold Masking - Differential Diffusion
If you want to switch to a threshold-based approach, change the code in the following file:

    /utilts/ddim.py | Lines 158-170


## Additional Resources
This ColorizeNet adaption is based on the [original ColorizeNet repository](https://github.com/rensortino/ColorizeNet) and an integration of the [Differential Diffusion approach](https://differential-diffusion.github.io/).
For more information on ControlNet, please refer to the [original repository](https://github.com/lllyasviel/ControlNet).


# Citation
    [Arxiv Link](https://arxiv.org/abs/2302.05543)

    @misc{zhang2023adding,
      title={Adding Conditional Control to Text-to-Image Diffusion Models}, 
      author={Lvmin Zhang and Anyi Rao and Maneesh Agrawala},
      booktitle={IEEE International Conference on Computer Vision (ICCV)}
      year={2023},
    }
***
    [Arxiv Link](https://arxiv.org/pdf/2306.00950)

    @misc{levin2024differentialdiffusiongivingpixel,
      title={Differential Diffusion: Giving Each Pixel Its Strength}, 
      author={Eran Levin and Ohad Fried},
      year={2024},
    }

    


