# Conditional Mammography Synthesis & Classification

This repository contains the synthesis and downstream classification pipelines used to study the utility of AI-generated mammograms for breast cancer classification.

The two main components are:

1. **Synthetic image generation** - `lightning_synthesis/`
2. **Classification using real and synthetic data** - primarily `holdout_train.py` and its supporting files in the repository root.

The overall workflow is:

```text
Real mammograms
      |
      v
lightning_synthesis/
Train conditional DDPM
      |
      v
Generate synthetic mammograms
      |
      +----------------+
      v                v
Real training data   Synthetic training data
      |                |
      +-------+--------+
              v
       holdout_train.py
              |
              v
      Classification model
              |
              v
   Evaluation on real test data
```

The repository supports training a class-conditional DDPM, generating synthetic mammograms with classifier-free guidance, training classifiers with different real/synthetic mixtures, optionally fine-tuning those classifiers on real data, and evaluating downstream classification performance.

## Repository Structure

The most important files are:

```text
U24_synthesis/
|
|-- config.yaml                         # Classification configuration
|-- holdout_train.py                    # Main classification training script
|-- data_loaders.py                     # Classification datasets/sampling
|-- models.py                           # Classification model definitions
|
|-- classification_test_evaluation.py
|-- classification_test_evaluation_MultiSeed.py
|-- 01_convnext_mulseed_eval.py
|
`-- lightning_synthesis/
    |-- config_l.yaml                   # Synthesis configuration
    |-- train.py                        # Main synthesis training script
    |-- data_loaders_l.py               # Synthesis datasets/sampling
    |-- inference.py                    # Synthetic image generation
    |-- inference_binary.py
    |-- evaluate.py                     # Synthesis evaluation
    |-- oracle_eval.py
    |
    `-- model_architectures/
        |-- monai_ddpm.py
        |-- noise_predictors.py
        `-- wcgan.py
```

These two pipelines are largely independent but share a common data organization: synthesis produces images in a structure that can subsequently be consumed by the classification pipeline.

# 1. Synthesis Pipeline

The synthesis code lives under `lightning_synthesis/`.

The main entry point is:

```bash
cd lightning_synthesis
python train.py
```

Training is configured through `config_l.yaml`.

`train.py` uses PyTorch Lightning and imports the synthesis dataset/data module from `data_loaders_l.py` and the generative models from `model_architectures/`.

Conceptually:

```text
config_l.yaml
      |
      v
data_loaders_l.py
      |
      v
train.py
      |
      v
model_architectures/
      |
      v
experiments/<experiment>/
```

## Synthesis Configuration

The main parameters in `config_l.yaml` include:

```yaml
root_dir: '/path/to/dataset/train'
data_dir: 'original'

num_classes: 2
model: DDPM

batch_size: 8
learning_rate: 0.00001
num_epochs: 50

dynamic_balanced_sampling: False
benign_to_malignant_ratio: 1.0

experiment_name: 'my_experiment'
```

For the binary experiments, the two classes are benign and malignant.

`dynamic_balanced_sampling` controls whether the class distribution is reconstructed dynamically during training. When enabled, `benign_to_malignant_ratio` determines the majority/minority sampling ratio.

## Synthesis Data Loading

`data_loaders_l.py` provides both the dataset implementation and Lightning data module.

The main components used by `train.py` are:

```python
NiftiSynthesisDataset
BalancedSamplingDataModule
```

The balanced data module can resample the majority class each epoch while retaining the minority class, allowing the benign:malignant training ratio to be controlled without permanently altering the underlying dataset.

## Generative Models

Model implementations live under:

```text
lightning_synthesis/model_architectures/
```

The synthesis training script currently imports:

```python
UNet
MonaiDDPM
ConditionalWGAN
```

The primary synthesis experiments use the conditional DDPM implementation.

## Generating Synthetic Images

After training a synthesis model, synthetic images are generated using the inference scripts:

```text
inference.py
inference_binary.py
```

Generated images are stored as NIfTI files. The generated dataset follows the same broad organization expected by the downstream classification pipeline:

```text
synthetic/
|-- benign/
|   |-- 0001/
|   |   `-- slice.nii.gz
|   |-- 0002/
|   |   `-- slice.nii.gz
|   `-- ...
|
`-- malignant/
    |-- 0001/
    |   `-- slice.nii.gz
    `-- ...
```

This allows generated data to be treated like another image source by `holdout_train.py`.

# 2. Classification Pipeline

The primary classification entry point is `holdout_train.py`.

It depends primarily on:

```text
config.yaml
data_loaders.py
models.py
```

The execution flow is:

```text
config.yaml
     |
     +--------------+
     v              v
data_loaders.py   models.py
     |              |
     +------+-------+
            v
     holdout_train.py
            |
            v
      experiments/
```

`holdout_train.py` handles dataset construction, train/validation splitting, real/synthetic mixing, augmentation, optional class balancing, model creation, training, validation, checkpointing, resuming, and fine-tuning.

## Classification Data Format

`NiftiDataset` expects data organized by class:

```text
<dataset>/
|-- benign/
|   |-- <sample_id>/
|   |   `-- image.nii.gz
|   `-- ...
|
`-- malignant/
    |-- <sample_id>/
    |   `-- image.nii.gz
    `-- ...
```

The class mapping is:

```text
benign       -> 0
malignant    -> 1
```

For each sample directory, the loader finds the first `.nii.gz` file, loads it using nibabel, converts it to a PyTorch tensor, adds a channel dimension, removes a trailing singleton dimension when present, and applies the configured transforms.

## Classification Configuration

Classification experiments are controlled through `config.yaml`.

Important settings include:

```yaml
root_dir: '/path/to/dataset/train'

data_dir: 'original'
synth_data_dir: 'synthetic_guide5.0'

num_classes: 2

real_percentage: 1.0
synthetic_multiplier: 0.75

dynamic_balanced_sampling: True
majority_to_minority_ratio: 1.0

batch_size: 32
learning_rate: 0.0001
num_epochs: 100

augmentations: geometric

model_name: convnext_tiny

early_stopping_metric: val_loss
early_stopping_patience: 10

seed: 44
```

Before starting an experiment, **check this file carefully**. Many experimental behaviors are controlled here rather than through command-line arguments.

## Running Classification Training

From the repository root:

```bash
python holdout_train.py
```

A seed can also be supplied as the first command-line argument:

```bash
python holdout_train.py 42
```

If no argument is provided, the seed in `config.yaml` is used.

## Real + Synthetic Training

Two parameters control the real/synthetic composition:

```yaml
real_percentage: 1.0
synthetic_multiplier: 0.75
```

These have different meanings.

`real_percentage` determines what fraction of the available **real training images in each class** is retained.

`synthetic_multiplier` determines how many synthetic images are added relative to the retained real samples.

For example:

```yaml
real_percentage: 1.0
synthetic_multiplier: 0.75
```

means:

```text
Keep 100% of eligible real training images

             +

Add synthetic images equal to approximately
75% of the retained real count, per class
```

Synthetic images are sampled without replacement. If fewer synthetic samples are available than requested, the code uses the available pool.

## Train / Validation Split

The holdout split is created **from the real dataset only** using a stratified `train_test_split`.

```text
Real dataset
     |
     +--------------> Real validation set
     |
     v
Real training subset
     |
     + Synthetic data
     |
     v
Training set
```

Synthetic images are therefore added to training, not validation.

The indices used for each experiment are saved to `indices.json`.

## Class Balancing vs. Synthetic Augmentation

These are **two separate mechanisms**.

```text
real_percentage
synthetic_multiplier
        |
        v
Real vs. synthetic composition
```

while:

```text
dynamic_balanced_sampling
majority_to_minority_ratio
        |
        v
Benign vs. malignant composition
```

When dynamic balancing is enabled, `make_balanced_loader()` rebuilds the training loader using the available training samples. It identifies the minority and majority classes, retains the minority samples, and samples the desired amount of the majority class.

## Classification Models

Classification architectures are defined in `models.py`.

The model factory currently implements:

```text
ResNet-50
EfficientNet-B0
ConvNeXt-Tiny
```

The networks are adapted for grayscale mammography.

The original first convolution expects three RGB channels. `adapt_first_conv_to_grayscale()` replaces it with a one-channel convolution and initializes its weights by averaging the pretrained RGB weights across the channel dimension.

The original classification layer is removed and replaced with:

```text
Backbone
   |
   v
BatchNorm
   |
   v
ReLU
   |
   v
Dropout
   |
   v
Linear classifier
```

## Training

The classifier uses:

```text
AdamW
CrossEntropyLoss with label smoothing
CosineAnnealingLR
```

The backbone receives a lower learning rate than the classification head.

Training alternates between `train` and `val` and records metrics including:

```text
Loss
Accuracy
ROC-AUC
Balanced Accuracy
Per-class Accuracy
```

The validation metric configured by `early_stopping_metric` determines whether a new checkpoint is saved.

## Experiment Outputs

Each classification run creates an experiment directory similar to:

```text
experiments/
`-- <experiment_name>/
    |-- config.yaml
    |-- models.py
    |-- holdout_train.py
    |-- data_loaders.py
    |-- indices.json
    |-- logs.csv
    `-- last.pt
```

The configuration and important source files are copied into the experiment directory when the experiment is first created. This provides a snapshot of the code/configuration used for that run.

If the experiment directory already exists, the training script treats the run as a resume attempt.

## Fine-Tuning

The classification pipeline also supports a second training stage.

Conceptually:

```text
Stage 1
Real + synthetic training
        |
        v
     last.pt
        |
        v
Stage 2
Real-only fine-tuning
```

Enable this through:

```yaml
fine_tune: True
fine_tune_ckpt: <previous experiment directory>
```

When fine-tuning is enabled, `holdout_train.py` forces:

```text
real_fraction = 1.0
synthetic_multiplier = 0.0
```

and loads the model weights from the selected previous experiment. The optimizer and scheduler are reinitialized rather than resumed.

## Classification Evaluation

Several root-level scripts are used for final evaluation.

`classification_test_evaluation.py` evaluates one classifier checkpoint.

`classification_test_evaluation_MultiSeed.py` evaluates multiple trained seeds and summarizes performance.

`01_convnext_mulseed_eval.py` compares experiment groups across seeds, including baseline versus synthetic-data conditions.

The main reported classification metrics are:

```text
Balanced Accuracy
Sensitivity
Specificity
ROC-AUC
```

For binary classification, malignant is treated as the positive class and benign as the negative class.

# Typical Experimental Workflow

For a new synthetic augmentation experiment:

```text
1. Configure synthesis
   lightning_synthesis/config_l.yaml

2. Train DDPM
   cd lightning_synthesis
   python train.py

3. Generate synthetic mammograms
   inference.py / inference_binary.py

4. Verify generated data organization

5. Return to repository root

6. Configure classification
   config.yaml

7. Train baseline and/or synthetic-augmented classifier
   python holdout_train.py <seed>

8. Optionally fine-tune on real data

9. Evaluate trained models
   classification_test_evaluation*.py

10. Compare results across seeds/conditions
```
