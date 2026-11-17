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
- [x] Change one factor at a time where possible; keep seeds, split, preprocessing, and evaluation protocol consistent.
- [x] Prioritize batch size / learning rate first; test 32 or 128 batch size as memory allows and learning rates such as `0.0001` versus `0.0002`.
- [ ] If training is stable and resources allow, compare `64 × 64` with `128 × 128`.
- [x] Choose a checkpoint using generated samples and evaluation scores together, not loss alone.

### Experiment log

Trial 1 findings: 50 epochs took 16.9 minutes (60,400 optimizer steps). Final epoch losses were D=0.1165 and G=6.2752. The loss plot, fixed-seed sample grid, checkpoint, and evaluation JSON are in `outputs/trial1_lr_d_1e4/`.

| Run | Change from baseline | Epochs | Best checkpoint | IS | FID | Visual notes |
|---|---|---:|---|---:|---:|---|
| Baseline | DCGAN defaults | 50 | `outputs/checkpoints/epoch_050.pt` | 2.525 | 0.0792 | Blurry painterly textures; repeated compositions; D appears dominant late |
| Trial 1 | D learning rate 0.0001; all else unchanged | 50 | `outputs/trial1_lr_d_1e4/checkpoints/epoch_050.pt` | 2.684 | 0.0372 | More varied painterly compositions; still blurry, but less repetitive than baseline |
| Trial 2 | D learning rate 0.00005; all else unchanged from Trial 1 | 50 | `outputs/trial2_lr_d_5e5/checkpoints/epoch_050.pt` | 2.951 | 0.0424 | Better IS than Trial 1; FID slightly worse; inspect grid alongside metrics |
| Trial 3 | Batch size 32; all other settings match Trial 2 | Running: 50 | `outputs/trial3_batch32/checkpoints/` (saving every 10 epochs) | Pending | Pending | Medium GPU run; metrics pending |
| Trial 4 | 2 generator updates per 1 discriminator update; all other settings match Trial 3 | Planned: 50 | Pending | Pending | Pending | Controlled update-ratio test |
| Trial 5 | Spectral normalization on D only; parent chosen from Trials 3-4 | Planned: 50 | Pending | Pending | Pending | Regularization/stability experiment |
| Trial 6 | Increase G width from 64 to 96 channels; D unchanged; parent selected by results | Planned: 50 | Pending | Pending | Pending | Generator capacity experiment |
| Trial 7 | Change G hidden activation from ReLU to LeakyReLU(0.2); parent selected by results | Planned: 50 | Pending | Pending | Pending | Activation experiment |
| Trial 8 | Add one residual block at 16x16 in G and D; optional if time allows | Planned: 50 | Pending | Pending | Pending | Depth experiment; implement only after earlier comparisons |

### Proposed follow-up experiments

Keep the 64x64 data, train/evaluation split, seed, latent size, optimizer settings, 50 epochs, and metric pipeline fixed unless a trial names that setting as its single change. Start each run from a fresh initialization with seed 42; do not fine-tune one trial's weights into another. Select the parent configuration for Trials 5-8 after comparing held-out FID/IS and fixed-seed sample grids, rather than automatically stacking every change.

1. **Trial 4 — update ratio.** Add a `generator_updates_per_discriminator` config setting, keep it at 1 for old trials, and set it to 2 for Trial 4. This gives G two optimizer steps for each D step. Keep all Trial 3 settings otherwise, so the update ratio is the only change. Trials 2-3 show low D loss and rising G loss; this is a targeted hypothesis, and the scores and sample grid decide whether it helps. Do not use the common WGAN-GP five-critic-step ratio with the current BCE DCGAN, because that belongs to a different objective and could make this discriminator stronger still.
2. **Trial 5 — discriminator spectral normalization.** Add spectral normalization to D's convolution layers only, leave the BCE-with-logits loss and G unchanged, and use the best parent from Trials 3-4. This regularizes D's layer gains and is a lower-scope architecture change than switching GAN objectives. Record any interaction with BatchNorm in D and the resulting losses/metrics. See Miyato et al., [Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957).
3. **Trial 6 — generator width.** Increase only `generator_channels` from 64 to 96; keep D at 64. This tests G capacity without simultaneously strengthening D. If memory or stability is poor, stop and use 80 channels as a follow-up rather than changing several settings at once.
4. **Trial 7 — generator activation.** Replace hidden G ReLU activations with LeakyReLU(0.2), keeping D's LeakyReLU and all widths fixed. Keep the final `tanh` output and image normalization unchanged. DCGAN's original architecture uses ReLU in G and LeakyReLU in D, so treat this as an empirical comparison rather than an assumed improvement; see [Radford et al.](https://arxiv.org/abs/1511.06434).
5. **Trial 8 — depth (optional).** Add one residual block at the 16x16 feature-map stage in both G and D, keeping widths fixed. This tests added depth as one matched architecture change. It will require a separate architecture version in checkpoints so evaluation reconstructs the correct model.

**Optional separate variant:** If the BCE DCGAN remains unstable after these controlled tests, consider a separate WGAN-GP implementation with critic loss, gradient penalty (start with lambda=10), no batch normalization in the critic, and its own documented update schedule. Do not compare it as a one-factor DCGAN tweak because it changes the training objective and critic constraints. The method is from Gulrajani et al., [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028).

**Expected resource cost:** Trial 3's current batch-32 pace suggests about 35-40 minutes including evaluation on this medium GPU. Trial 4's two G updates per D update should take roughly 45-55 minutes including evaluation. Record actual times and check storage after its first checkpoint.

### Trial 2 pre-run record

- Config: `config_trial2.json`; output directory: `outputs/trial2_lr_d_5e5/`.
- Hypothesis: reducing D's learning rate from 1e-4 to 5e-5 may further reduce discriminator dominance and improve sample diversity/metrics; it may also weaken D too much.
- Controlled variables: seed 42, same sorted-file 95/5 split, 64x64, batch 64, 50 epochs, latent dimension 100, G learning rate 2e-4, Adam betas (0.5, 0.999), no horizontal flip, and the same Ignite evaluation protocol.
- Submitted: 2026-11-15 16:11 PST. The scheduler queued the pod for about 79 minutes, then assigned it to `its-dsmlp-n05.ucsd.edu` at 17:30 PST.
- Launch outcome: The interactive DSMLP stream failed with `tls: bad record MAC` immediately after the container became active; DSMLP deleted pod `xix015-17520` before a training command was run. This was a launch failure, not a Trial 2 training result.
- Retry on 2026-11-16: SSH reached the UCSD Duo notice but the connection reset before launch. No Trial 2 training has run yet.
- Retry 2 on 2026-11-16 09:35 PST: Duo authentication succeeded; DSMLP submitted job `xix015-1982700`. It remained queued because both eligible GPU slots were occupied, then was canceled at 09:58 PST at the user's request so a DataHub-launched session can be used instead.
- Batch-mode attempt on 2026-11-16 10:08 PST: `-B` submitted pod `xix015-2010299` and returned without waiting in the foreground. The batch command then stopped before training because the container did not have the old `../.venv/bin/activate` path. No training ran in this attempt. Use `run_trial2_batch.sh` to create/reuse a venv inside the container and verify CUDA before training.
- Batch-mode retry on 2026-11-16: pod `xix015-2013530` was rejected by the admission webhook because the completed pod from the earlier attempt still counted against the GPU quota. The completed pod was deleted; no training ran.
- GPU targeting check on 2026-11-16: a dry run with the intended ECE285 course workspace and `2025.1-stable` image produced the correct resource request. An A5000-specific batch pod (`xix015-2025586`) was then submitted, but the scheduler found no node matching both `gpu-class=small` and `gputype=a5000`; it was deleted. The status page showed some A5000 capacity cluster-wide, but that did not make it eligible under the small-GPU selector.
- Generic small-GPU batch on 2026-11-16 10:24 PST: submitted `xix015-2027169` with `-B -g 1 -c 8 -m 16 -W ECE285_FA26_002`, the course's `2025.1-stable` image, and `run_trial2_batch.sh`. It initially waited with 2 eligible small-GPU nodes lacking capacity, then the scheduler placed it on `its-dsmlp-n06.ucsd.edu` after about 7 minutes. The script exited before training because the repository is mounted in the container under `$HOME/private/285hw2-gan/project`; its earlier path was wrong. This confirms the generic small-GPU request can be scheduled even though the status-page snapshot showed the 1080 Ti nodes full.
- CPU-only workspace diagnostic on 2026-11-16 10:40 PST: confirmed the repo and batch script are visible at `/home/xix015/private/285hw2-gan/project` inside a course pod. Corrected `run_trial2_batch.sh` to use this mounted path before resubmitting Trial 2.
- Corrected small-GPU retry on 2026-11-16 10:42 PST: removed the failed and diagnostic pods, pulled the path fix from GitHub, and submitted `xix015-2048363` with the same generic small-GPU request. It remained Pending after 32 minutes with two matching nodes reporting insufficient GPU capacity. Canceled this Trial 2 GPU wait when switching Trial 2 to CPU.
- Concurrent run plan: Trial 2 remains 50 epochs on CPU, with four CPU cores; Trial 3 uses one generic small GPU and changes only batch size from 64 to 32 (`config_trial3.json`, output `outputs/trial3_batch32/`). Trial 3's batch-size setting was selected before Trial 2 results were available so its GPU request could queue in parallel.
- First CPU attempt on 2026-11-16 11:19 PST: pod `xix015-2086399` started on CPU node `its-dsmlp-n28.ucsd.edu`, confirmed `CUDA available: False`, then stopped before training because the configured dataset symlink resolved to an unmounted DSMLP login-node path. No training ran. The pod was deleted after collecting logs.
- Workspace diagnostic on 2026-11-16: confirmed the dataset is visible inside pods at `/home/xix015/private/285hw2-gan/wikiart/versions/1`. The queued first Trial 3 pod `xix015-2087049` was deleted before scheduling to avoid running its stale path configuration.
- Updated both launch scripts to generate runtime configs with container-visible absolute dataset/output paths and separate per-trial virtual environments, avoiding concurrent pip changes to the same environment.
- CPU run plan: Trial 2 used 4 CPUs and 16 GB RAM while GPU capacity was unavailable; this attempt was stopped after the account quota blocked log/checkpoint writes.
- First corrected CPU submission on 2026-11-16 11:25 PST: pod `xix015-2093010` reached CPU node `its-dsmlp-n28.ucsd.edu`, passed the container-path check, and started `train.py --device cpu`.
- Trial 3 GPU submission on 2026-11-16 11:26 PST: pod `xix015-2094005` was submitted with the generic small-GPU selector, 4 CPU cores/16 GB RAM, and `run_trial3_batch.sh`. It was initially Pending; the scheduler later placed it on a matching node.
- Trial 2 CPU progress check on 2026-11-16 12:16 PST: after 43.3 minutes, 3/50 epochs were complete. Epoch losses: (1) D=1.7212, G=1.5513; (2) D=1.3424, G=0.8553; (3) D=1.3957, G=0.7743. Estimated 50-epoch runtime was about 12 hours before evaluation.
- Disk and CPU progress check on 2026-11-16: Trial 2 reached epoch 13/50 (D=0.8985, G=1.6585; elapsed 191 minutes). The account's low-space notification reported 481 MB remaining; the mounted filesystem's `df` free-space figure did not report the personal quota.
- Trial 3 GPU outcome on 2026-11-16: the generic small-GPU pod `xix015-2094005` was assigned to `its-dsmlp-n06.ucsd.edu` (GTX 1080 Ti) after about 14 minutes queued, then failed before the first training step. The installed user-site PyTorch 2.9.1+cu128 supports only `sm_70+`, but this GPU is `sm_61` (`no kernel image is available`). The failed pod was deleted. The DSMLP image's base stack is PyTorch 2.2.1+cu121 / torchvision 0.17.1+cu121; Trial 3's launcher now disables user-site packages and checks a CUDA kernel before training.
- CPU Trial 2 outcome: the run reached epoch 14/50, then failed with `OSError: [Errno 122] Disk quota exceeded` while writing training output. It was stopped and not resumed; its partial checkpoint files (about 1.1 GB total) and its GAN-specific CPU virtual environment (about 25 MB) were removed to free account storage. Dataset, source, and earlier successful GAN results were retained.
- Training start time / duration: medium-GPU Trial 2 restarted on 2026-11-16 at about 19:13 PST in pod `xix015-2644200` on `its-dsmlp-n54.ucsd.edu`.
- Medium-GPU restart: `gpu-class=medium`, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition; 4 CPU cores and 16 GB RAM. Training used CUDA and passed the CUDA kernel check. Checkpoints were configured every 10 epochs to limit disk use.
- Medium-GPU training result: 50/50 epochs finished in 16.8 minutes. Final losses: D=0.2926, G=4.2603. Epoch 10, 20, 30, 40, and 50 checkpoints saved successfully; output directory reached about 454 MB, with no quota error.
- Evaluation result: Ignite 0.5.5 on `cuda:0`; 4,073 held-out real images and 4,073 generated images resized to 299x299; IS=2.950748, FID=0.042358. Results are in `outputs/trial2_lr_d_5e5/evaluation.json`.
- Comparison: Trial 2 improves IS over Trial 1 (2.684 to 2.951), while FID is slightly higher (0.0372 to 0.0424); both FID scores remain below the baseline's 0.0792. Trial 1 remains the best FID run so far. Review the generated grid before making a visual quality claim.
- Cleanup/storage follow-up: deleting the failed CPU run's partial checkpoints freed about 1.1 GB. Medium Trial 2 then wrote its checkpoint set and evaluation output successfully. Preserve the dataset and prior GAN results; use a 10-epoch checkpoint interval for Trial 3 as well.
- Trial 3 compatibility retry on 2026-11-16 12:22 PST: pod `xix015-2155027` used the image's base PyTorch 2.2.1+cu121 / torchvision 0.17.1+cu121, with user-site packages disabled and a CUDA kernel preflight. That generic small-GPU attempt is obsolete for medium Blackwell, which requires the newer CUDA-enabled PyTorch environment used by Trial 2.
- Trial 3 medium-GPU run launched 2026-11-16 about 19:48 PST as pod `xix015-2693436` on `its-dsmlp-n54.ucsd.edu`. It uses batch size 32, Trial 2's D learning rate (5e-5), all other Trial 2 settings unchanged, and checkpoints every 10 epochs. PyTorch 2.9.1+cu128 / torchvision 0.24.1+cu128 passed the CUDA kernel check on the RTX PRO 6000 Blackwell MIG (capability 12.0). Initial progress: epoch 1 D=1.3953, G=1.4562; output directory 2.8 MB and Trial 3 venv 20 MB.
- Trial 3 progress check (2026-11-16 about 20:02 PST): 28/50 epochs; epoch 28 D=0.1861, G=5.1025. Epoch 10 and 20 checkpoints saved; output directory was about 224 MB shortly before this check. Training and metric evaluation are still running.
- Trial 4 implementation is prepared locally: `train.py` now supports `generator_updates_per_discriminator` (defaults to 1), and `config_trial4.json` / `run_trial4_batch.sh` set it to 2 with all other settings copied from Trial 3. Training logs report the mean G loss across its two updates for fair per-batch comparisons. Do not launch until Trial 3 finishes and its metrics/grid are recorded.
- Current status: Trial 2 training and evaluation are complete. Trial 3 is running on medium GPU pod `xix015-2693436`; monitor its checkpoint writes and evaluation results.

## 6. Generate final samples and evaluate

- [x] Generate a fixed number of images from the selected checkpoint; save them under a dedicated output folder.
- [x] Create a grid of representative samples for the report, using a fixed random seed and no cherry-picking beyond clearly documenting the selection.
- [x] Compute IS on generated images.
- [x] Compute FID between generated images and the held-out real paintings (not the training split).
- [x] Keep the metric implementation, preprocessing, image count, and resize policy identical across runs. Record Ignite/PyTorch versions.
- [x] Report the number of real and generated images used. For Inception-based metrics, follow the selected implementation's input-size and value-range requirements.

### Evaluation results

- Selected model/checkpoint: Trial 1 `outputs/trial1_lr_d_1e4/checkpoints/epoch_050.pt` (chosen by improved held-out scores and visual comparison).
- Metric implementation and versions: PyTorch-Ignite 0.5.5; PyTorch 2.9.1+cu128; torchvision 0.24.1+cu128.
- Real evaluation images: 4,073 held-out images.
- Generated images: 4,073 for metrics; 64 for the saved visual sample set.
- Resize / normalization details: model output/real images converted to RGB `[0,1]`, resized to 299×299 for Ignite Inception metrics.
- Inception Score (mean ± std, if available): 2.684 (Ignite returned a scalar for this run).
- FID: 0.0372 (Ignite default torchvision Inception/logit feature implementation).
- Qualitative observations: Trial 1 improves on baseline IS (2.525) and FID (0.0792) under the same protocol; its samples show more varied painterly compositions, with blur and artifacts remaining.
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
