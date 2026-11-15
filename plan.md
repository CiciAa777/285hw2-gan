# ECE 285 HW2: WikiArt GAN Plan

## Goal

Train an unconditional GAN on the downloaded WikiArt paintings, generate sample paintings, evaluate the model with Inception Score (IS) and Fréchet Inception Distance (FID), and write a two-page NeurIPS-format report.

## 1. Inspect the dataset

- [ ] Confirm the dataset root. Current expected path: `hw2/steubk/wikiart/versions/1/`.
- [ ] Count images by style folder and check for unreadable or non-image files.
- [ ] Make a contact sheet of random paintings to inspect image quality, aspect ratios, and color modes.
- [ ] Decide whether to use every style folder. Default: use all style folders for one unconditional model; folder labels are not model inputs.
- [ ] Record the dataset split seed and held-out evaluation set size.

### Dataset findings

- Dataset root:
- Image count after filtering:
- Number of style folders used:
- Images per folder (range / notable imbalance):
- Corrupt or skipped files:
- Example aspect ratios / image issues:
- Train and evaluation split:
- Other notes:

## 2. Prepare the image pipeline

- [ ] Build a recursive image dataset/loader rooted at the dataset directory. Folder labels can be ignored for unconditional training.
- [ ] Convert all images to RGB.
- [ ] Resize/crop to a consistent square resolution. Start with `64 × 64`; consider `128 × 128` only if compute allows.
- [ ] Normalize channels to `[-1, 1]` to match a generator output layer using `tanh`.
- [ ] Decide on augmentation. Start with random horizontal flip only if it does not harm the painting content; keep evaluation preprocessing deterministic.
- [ ] Preview a batch after preprocessing before training.

### Preprocessing findings

- Resolution:
- Resize/crop policy:
- Augmentations:
- Normalization:
- Batch preview observations:
- Changes needed:

## 3. Implement a DCGAN baseline

- [ ] Implement a convolutional generator: latent noise → learned projection → upsampling/convolution blocks → RGB image.
- [ ] Implement a convolutional discriminator: RGB image → downsampling/convolution blocks → real/fake logit.
- [ ] Use batch normalization in the generator and a `tanh` output for normalized images. Use a sigmoid-free discriminator output when using logits-based binary cross-entropy.
- [ ] Add DCGAN-style weight initialization and fixed latent vectors for consistent sample grids.
- [ ] Save model configuration and checkpoints with the run outputs.

### Baseline architecture notes

- Generator block/channel design:
- Discriminator block/channel design:
- Latent dimension:
- Loss function:
- Initialization / normalization details:
- Implementation issues:

## 4. Train the baseline

- [ ] Set and record random seeds and hardware/software versions.
- [ ] Train on the training split. For each step, update the discriminator using real and generated images, then update the generator to fool the discriminator.
- [ ] Log generator and discriminator losses at every iteration or at a documented interval.
- [ ] Save fixed-noise sample grids regularly (for example, once per epoch) and save checkpoints regularly.
- [ ] Keep the best checkpoint and final checkpoint; record which one is used for evaluation.
- [ ] Stop or adjust training if samples collapse, become repetitive, or fail to improve.

### Training configuration

| Hyperparameter | Initial value | Final value / notes |
|---|---:|---|
| Image size | 64 × 64 | |
| Batch size | 64 | |
| Latent dimension | 100 | |
| Generator learning rate | 0.0002 | |
| Discriminator learning rate | 0.0002 | |
| Adam betas | (0.5, 0.999) | |
| Epochs |  | |
| D updates per G update | 1 | |
| Seed |  | |
| Hardware |  | |

### Training findings

- Run/checkpoint used:
- Training duration:
- Loss-curve observations:
- Sample-quality progression:
- Signs of instability or mode collapse:
- Other notes:

## 5. Tune a small number of settings

- [ ] Establish the baseline before tuning.
- [ ] Change one factor at a time where possible; keep seeds, split, preprocessing, and evaluation protocol consistent.
- [ ] Prioritize batch size / learning rate first; test 32 or 128 batch size as memory allows and learning rates such as `0.0001` versus `0.0002`.
- [ ] If training is stable and resources allow, compare `64 × 64` with `128 × 128`.
- [ ] Choose a checkpoint using generated samples and evaluation scores together, not loss alone.

### Experiment log

| Run | Change from baseline | Epochs | Best checkpoint | IS | FID | Visual notes |
|---|---|---:|---|---:|---:|---|
| Baseline | | | | | | |
| Trial 1 | | | | | | |
| Trial 2 | | | | | | |
| Trial 3 | | | | | | |

## 6. Generate final samples and evaluate

- [ ] Generate a fixed number of images from the selected checkpoint; save them under a dedicated output folder.
- [ ] Create a grid of representative samples for the report, using a fixed random seed and no cherry-picking beyond clearly documenting the selection.
- [ ] Compute IS on generated images.
- [ ] Compute FID between generated images and the held-out real paintings (not the training split).
- [ ] Keep the metric implementation, preprocessing, image count, and resize policy identical across runs. Record Ignite/PyTorch versions.
- [ ] Report the number of real and generated images used. For Inception-based metrics, follow the selected implementation's input-size and value-range requirements.

### Evaluation results

- Selected model/checkpoint:
- Metric implementation and versions:
- Real evaluation images:
- Generated images:
- Resize / normalization details:
- Inception Score (mean ± std, if available):
- FID:
- Qualitative observations:
- Limitations / caveats:

## 7. Write the two-page report

- [ ] Use the required NeurIPS format and stay within two pages (check whether references count toward the limit).
- [ ] Suggested structure: motivation/data; model and training setup; results; limitations/conclusion.
- [ ] Include a compact hyperparameter table, generator/discriminator loss plot, generated-image grid, and IS/FID results.
- [ ] Caption figures with resolution, checkpoint/run, sample count, and evaluation protocol as relevant.
- [ ] Cite the WikiArt dataset and the metric implementation/tutorial.
- [ ] Verify that all reported values match saved logs and outputs.

### Report findings and results

- Main result:
- Best configuration and why:
- What the generated images capture well:
- Common artifacts or failure modes:
- What the scores suggest (and do not establish):
- Limitations:
- Follow-up experiments:

## Final checklist

- [ ] Training script/config and reproducibility details are saved.
- [ ] Loss curves and final sample grid are saved.
- [ ] IS/FID scores and evaluation protocol are recorded.
- [ ] Final checkpoint is saved and identified.
- [ ] Two-page report is complete and figures are readable.
