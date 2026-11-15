# ECE 285 HW2: WikiArt GAN Plan

## Goal

Train an unconditional GAN on the downloaded WikiArt paintings, generate sample paintings, evaluate the model with Inception Score (IS) and Fréchet Inception Distance (FID), and write a two-page NeurIPS-format report.

## 1. Inspect the dataset

- [x] Confirm the dataset root. Current expected path: `hw2/steubk/wikiart/versions/1/`.
- [x] Count images by style folder and check for unreadable or non-image files.
- [x] Make a contact sheet of random paintings to inspect image quality, aspect ratios, and color modes.
- [x] Decide whether to use every style folder. Use all style folders for one unconditional model; folder labels are not model inputs.
- [x] Record the dataset split seed and held-out evaluation set size.

### Dataset findings

- Dataset root: `/home/xix015/285hw2-gan/steubk/wikiart/versions/1` (symlink to persistent DSMLP personal storage).
- Image count after filtering: 81,444 valid 64×64 JPEGs.
- Number of style folders used: 27.
- Images per folder (range / notable imbalance): 98–13,060; notable class imbalance, but styles are pooled for unconditional training.
- Corrupt or skipped files: 0 corrupt; 81,444 discovered and valid.
- Example aspect ratios / image issues: images were center-cropped/resized to square 64×64 before transfer; source aspect ratios are not retained in the DSMLP copy.
- Train and evaluation split: seed 42, 77,371 train and 4,073 held out (95% / 5%).
- Other notes: the 31.4 GB full-resolution archive exceeded DSMLP course-home quota; all paintings are included at model input resolution.

## 2. Prepare the image pipeline

- [x] Build a recursive image dataset/loader rooted at the dataset directory. Folder labels can be ignored for unconditional training.
- [x] Convert all images to RGB.
- [x] Resize/crop to a consistent square resolution. Start with `64 × 64`; consider `128 × 128` only if compute allows.
- [x] Normalize channels to `[-1, 1]` to match a generator output layer using `tanh`.
- [x] Decide on augmentation. Start with random horizontal flip only if it does not harm the painting content; keep evaluation preprocessing deterministic.
- [x] Preview a batch after preprocessing before training.

### Preprocessing findings

- Resolution: 64 × 64.
- Resize/crop policy: source paintings center-cropped/resized to square on local machine; loader also applies deterministic square fit and converts RGB.
- Augmentations: disabled.
- Normalization: `[-1, 1]`.
- Batch preview observations: smoke batch shape `(64, 3, 64, 64)`, values within `[-1, 1]`.
- Changes needed:

## 3. Implement a DCGAN baseline

- [x] Implement a convolutional generator: latent noise → learned projection → upsampling/convolution blocks → RGB image.
- [x] Implement a convolutional discriminator: RGB image → downsampling/convolution blocks → real/fake logit.
- [x] Use batch normalization in the generator and a `tanh` output for normalized images. Use a sigmoid-free discriminator output when using logits-based binary cross-entropy.
- [x] Add DCGAN-style weight initialization and fixed latent vectors for consistent sample grids.
- [x] Save model configuration and checkpoints with the run outputs.

### Baseline architecture notes

- Generator block/channel design:
- Discriminator block/channel design:
- Latent dimension:
- Loss function:
- Initialization / normalization details:
- Implementation issues: smoke test confirmed finite losses, checkpoint reload, and 64×64 sample-grid generation. The course image's base PyTorch did not support its Blackwell MIG GPU; matched PyTorch 2.9.1+cu128 and torchvision 0.24.1+cu128 are installed in the user's DSMLP environment. Ignite IS/FID batch updates pass.

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
| Epochs | 50 | |
| D updates per G update | 1 | |
| Seed | 42 | |
| Hardware | ECE285 medium GPU, 24 GB MIG slice | |

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
