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

- Generator block/channel design: 4×4 512 channels, then transposed-convolution upsampling through 256, 128, 64 channels to 64×64 RGB; batch normalization/ReLU, final `tanh`.
- Discriminator block/channel design: 64×64 RGB downsampling through 64, 128, 256, 512 channels to one real/fake logit; LeakyReLU, batch normalization after the first stage.
- Latent dimension: 100.
- Loss function: `BCEWithLogitsLoss` for both networks; one D update per G update.
- Initialization / normalization details: DCGAN normal initialization (`std=0.02`; BN weights centered at 1); input RGB scaled to `[-1,1]`.
- Implementation issues: smoke test confirmed finite losses, checkpoint reload, and 64×64 sample-grid generation. The course image's base PyTorch did not support its Blackwell MIG GPU; matched PyTorch 2.9.1+cu128 and torchvision 0.24.1+cu128 are installed in the user's DSMLP environment. Ignite IS/FID batch updates pass.

## 4. Train the baseline

- [x] Set and record random seeds and hardware/software versions.
- [x] Train on the training split. For each step, update the discriminator using real and generated images, then update the generator to fool the discriminator.
- [x] Log generator and discriminator losses at every iteration or at a documented interval.
- [x] Save fixed-noise sample grids regularly (for example, once per epoch) and save checkpoints regularly.
- [x] Keep the best checkpoint and final checkpoint; record which one is used for evaluation.
- [x] Stop or adjust training if samples collapse, become repetitive, or fail to improve.

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
| Hardware | ECE285 medium GPU: RTX PRO 6000 Blackwell, 1g.24gb MIG | |

### Training findings

- Run/checkpoint used: baseline, `outputs/checkpoints/epoch_050.pt` (`latest.pt` is the same epoch).
- Training duration: 16.8 minutes for 50 epochs; 60,400 optimizer steps.
- Loss-curve observations: D loss fell from 0.5912 at epoch 1 to 0.0816 at epoch 50; G loss was 5.5244 at epoch 1 and 7.7006 at epoch 50, with sustained high variance and a late rise.
- Sample-quality progression: the epoch-50 grid shows recognizable painting-like color fields and textures, though images are blurry and some compositions repeat.
- Signs of instability or mode collapse: discriminator appears to dominate late in training (low D loss, high G loss); the sample grid has some repetition but not complete collapse.
- Other notes: `outputs/logs/losses.csv` records all 60,400 steps; `outputs/plots/loss_curves.png` and `outputs/generated/epoch_050/grid.png` are saved. Resume support was smoke-tested across an epoch boundary.

## 5. Tune a small number of settings

- [x] Establish the baseline before tuning.
- [ ] Change one factor at a time where possible; keep seeds, split, preprocessing, and evaluation protocol consistent.
- [ ] Prioritize batch size / learning rate first; test 32 or 128 batch size as memory allows and learning rates such as `0.0001` versus `0.0002`.
- [ ] If training is stable and resources allow, compare `64 × 64` with `128 × 128`.
- [ ] Choose a checkpoint using generated samples and evaluation scores together, not loss alone.

### Experiment log

| Run | Change from baseline | Epochs | Best checkpoint | IS | FID | Visual notes |
|---|---|---:|---|---:|---:|---|
| Baseline | DCGAN defaults | 50 | `outputs/checkpoints/epoch_050.pt` | 2.525 | 0.0792 | Blurry painterly textures; repeated compositions; D appears dominant late |
| Trial 1 | Planned: D learning rate 0.0001; all else unchanged | 50 | | | | |
| Trial 2 | | | | | | |
| Trial 3 | | | | | | |

## 6. Generate final samples and evaluate

- [x] Generate a fixed number of images from the selected checkpoint; save them under a dedicated output folder.
- [x] Create a grid of representative samples for the report, using a fixed random seed and no cherry-picking beyond clearly documenting the selection.
- [x] Compute IS on generated images.
- [x] Compute FID between generated images and the held-out real paintings (not the training split).
- [x] Keep the metric implementation, preprocessing, image count, and resize policy identical across runs. Record Ignite/PyTorch versions.
- [x] Report the number of real and generated images used. For Inception-based metrics, follow the selected implementation's input-size and value-range requirements.

### Evaluation results

- Selected model/checkpoint: baseline `outputs/checkpoints/epoch_050.pt`.
- Metric implementation and versions: PyTorch-Ignite 0.5.5; PyTorch 2.9.1+cu128; torchvision 0.24.1+cu128.
- Real evaluation images: 4,073 held-out images.
- Generated images: 4,073 for metrics; 64 for the saved visual sample set.
- Resize / normalization details: model output/real images converted to RGB `[0,1]`, resized to 299×299 for Ignite Inception metrics.
- Inception Score (mean ± std, if available): 2.525 (Ignite returned a scalar for this run).
- FID: 0.0792 (Ignite default torchvision Inception/logit feature implementation).
- Qualitative observations: metric values are recorded for comparisons under this exact implementation/protocol; visual quality is still blurry and repetitive in places.
- Limitations / caveats: Ignite's default FID implementation differs from `pytorch-fid`; do not compare the numeric value directly with scores from other FID implementations. The 64×64 input resolution limits fine detail.

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
