# SAM: Sharpness-Aware Minimization for Efficiently Improving Generalization

by Pierre Foret, Ariel Kleiner, Hossein Mobahi and Behnam Neyshabur.


## SAM in a few words

**Abstract**: In today's heavily overparameterized models, the value of the training loss provides few guarantees on model generalization ability. Indeed, optimizing only the training loss value, as is commonly done, can easily lead to suboptimal model quality. Motivated by the connection between geometry of the loss landscape and generalization---including a generalization bound that we prove here---we introduce a novel, effective procedure for instead simultaneously minimizing loss value and loss sharpness. In particular, our procedure, Sharpness-Aware Minimization (SAM), seeks parameters that lie in neighborhoods having uniformly low loss; this formulation results in a min-max optimization problem on which gradient descent can be performed efficiently. We present empirical results showing that SAM improves model generalization across a variety of benchmark datasets (e.g., CIFAR-{10, 100}, ImageNet, finetuning tasks) and models, yielding novel state-of-the-art performance for several. Additionally, we find that SAM natively provides robustness to label noise on par with that provided by state-of-the-art procedures that specifically target learning with noisy labels.


|      ![fig](figures/summary_plot.png)    |  ![fig](figures/no_sam.png)   |  ![fig](figures/sam_wide.png)     |
|:--------------:|:----------:|:----------------------:|
| Error rate reduction obtained by switching to SAM. Each point is a different dataset / model / data augmentation | A sharp minimum to which a ResNet trained with SGD converged | A wide minimum to which the same ResNet trained with SAM converged. |



## About this repo

This code allows the user to replicate most of the experiments of the paper, including:

 * Training from scratch Wideresnets and Pyramidnets (with shake shake / shake drop) on CIFAR10/CIFAR100/SVHN/Fashion MNIST, with or without SAM, with or without cutout and AutoAugment.
 * Training Resnets and Efficientnet on Imagenet, with or without SAM or RandAugment.
 * Finetuning Efficientnet from checkpoints trained on Imagenet/JFT on imagenet.


## How to train from scratch

Once the repo is cloned, experiments can be launched using sam.sam_jax.train.py:

```
python3 -m sam.sam_jax.train --dataset cifar10 --model_name WideResnet28x10 \
--output_dir /tmp/my_experiment --image_level_augmentations autoaugment \
--num_epochs 1800 --sam_rho 0.05
```

Note that our code uses all available GPUs/TPUs for training.

To see a detailed list of all available flags, run python3 -m sam.sam_jax.train --help.

#### Output

Training curves can be loaded using TensorBoard. TensorBoard events will be
saved in the output_dir, and their path will contain the learning_rate,
the weight_decay, rho and the random_seed.

## Finetuning EfficientNet:

We provide a FLAX checkpoint compatible with our implementation for all checkpoints
available on [official tensorflow efficientnet implementation](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet):

|               |   B0    |  B1   |  B2    |  B3   |  B4   |  B5    | B6 | B7 | B8 |  L2 |
|----------     |--------  | ------| ------|------ |------ |------ | --- | --- | --- | --- |
| Baseline preprocessing |  76.7% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/ckpts/efficientnet-b0/checkpoint.tar.gz))   | 78.7% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/ckpts/efficientnet-b1/checkpoint.tar.gz))  | 79.8% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/ckpts/efficientnet-b2/checkpoint.tar.gz)) | 81.1% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/ckpts/efficientnet-b3/checkpoint.tar.gz)) | 82.5% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/ckpts/efficientnet-b4/checkpoint.tar.gz)) | 83.1% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/ckpts/efficientnet-b5/checkpoint.tar.gz)) | | || | |
| AutoAugment (AA) |  77.1% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/ckptsaug/efficientnet-b0/checkpoint.tar.gz))   | 79.1% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/ckptsaug/efficientnet-b1/checkpoint.tar.gz))  | 80.1% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/ckptsaug/efficientnet-b2/checkpoint.tar.gz)) | 81.6% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/ckptsaug/efficientnet-b3/checkpoint.tar.gz)) | 82.9% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/ckptsaug/efficientnet-b4/checkpoint.tar.gz)) | 83.6% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/ckptsaug/efficientnet-b5/checkpoint.tar.gz)) |  84.0% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/ckptsaug/efficientnet-b6/checkpoint.tar.gz)) | 84.3% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/ckptsaug/efficientnet-b7/checkpoint.tar.gz))  || |
| RandAugment (RA) |  |  |  |  |  | 83.7%  ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/randaug/efficientnet-b5-randaug/checkpoint.tar.gz)) |  | 84.7%  ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/randaug/efficientnet-b7-randaug/checkpoint.tar.gz)) |  | |
| AdvProp + AA | 77.6% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/advprop/efficientnet-b0/checkpoint.tar.gz)) | 79.6% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/advprop/efficientnet-b1/checkpoint.tar.gz))  | 80.5% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/advprop/efficientnet-b2/checkpoint.tar.gz)) | 81.9% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/advprop/efficientnet-b3/checkpoint.tar.gz)) | 83.3% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/advprop/efficientnet-b4/checkpoint.tar.gz)) | 84.3% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/advprop/efficientnet-b5/checkpoint.tar.gz)) | 84.8% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/advprop/efficientnet-b6/checkpoint.tar.gz)) | 85.2% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/advprop/efficientnet-b7/checkpoint.tar.gz)) | 85.5% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/advprop/efficientnet-b8/checkpoint.tar.gz))|| |
| NoisyStudent + RA | 78.8% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/noisystudent/efficientnet-b0/checkpoint.tar.gz)) | 81.5% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/noisystudent/efficientnet-b1/checkpoint.tar.gz)) | 82.4% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/noisystudent/efficientnet-b2/checkpoint.tar.gz)) | 84.1% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/noisystudent/efficientnet-b3/checkpoint.tar.gz)) | 85.3% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/noisystudent/efficientnet-b4/checkpoint.tar.gz)) | 86.1% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/noisystudent/efficientnet-b5/checkpoint.tar.gz)) | 86.4% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/noisystudent/efficientnet-b6/checkpoint.tar.gz)) | 86.9% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/noisystudent/efficientnet-b7/checkpoint.tar.gz)) | - |88.4% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/noisystudent/efficientnet-l2/checkpoint.tar.gz)) |

<sup>* We report in this table the scores as found [here](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet). If you use these checkpoints with another codebase, you might obtain some slightly different scores based on the type of accelerator you use and your data processing pipeline (resizing algorithm in particular).</sup>

<sup>* Advprop requires some slight modification of the input pipeline. See [here](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) for more details. </sup>

<sup>* Please refer to the README of the [official tensorflow efficientnet implementation](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) to see which paper should be cited for which checkpoint.</sup>

Once the checkpoint is downloaded and uncompressed, it can be finetuned on any of the datasets:

```
python3 -m sam.sam_jax.train --output_dir /tmp/my_finetuning_experiment \
--dataset imagenet --from_pretrained_checkpoint true \
--efficientnet_checkpoint_path /tmp/path_to_efficientnet_checkpoint_to_finetune \
--learning_rate 0.1 --model_name efficientnet-l2-475 --batch_size 512 \
--num_epochs 10 --gradient_clipping 1.0 --label_smoothing 0.1 --sam_rho 0.1
```


## Bibtex

```
@ARTICLE{2020arXiv201001412F,
       author = {{Foret}, Pierre and {Kleiner}, Ariel and {Mobahi}, Hossein and {Neyshabur}, Behnam},
        title = "{Sharpness-Aware Minimization for Efficiently Improving Generalization}",
         year = 2020,
          eid = {arXiv:2010.01412},
       eprint = {2010.01412},
}
```

**This is not an official Google product.**
