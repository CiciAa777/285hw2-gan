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
| Trial 3 | Batch size 32; all other settings match Trial 2 | 50 | Removed after archiving evidence to `results/trial3_batch32/` | 2.118 | 0.1165 | Repetitive green/gray motifs; worse than Trial 2 |
| Trial 4 | 2 generator updates per 1 discriminator update; all other settings match Trial 3 | 50 | Removed after archiving evidence to `results/trial4_g2d1/` | 2.807 | 0.1631 | Higher IS than Trial 3, but worse FID; several repeated motifs |
| Trial 5 | Spectral normalization on D only; same settings as Trial 2 | 50 | Archived in `results/trial5_sn_d/`; remote checkpoint removed after archival | 2.295 | 0.0987 | Better than Trial 3 on both metrics; below Trials 1-2; repeated vertical white forms |
| Trial 6 | Increase G width from 64 to 96 channels; D unchanged from Trial 2 | 50 | Archived in `results/trial6_g_width96/` | 2.937 | 0.0419 | Essentially tied with Trial 2; no clear gain from wider G |
| Trial 7 | Change G hidden activation from ReLU to LeakyReLU(0.2); Trial 2 settings | Stopped at epoch 4; no checkpoint | No completed run | N/A | N/A | Interrupted by user before completion; no evaluation |
| Trial 8 | Add one residual block at 16x16 in G and D; optional if time allows | Planned: 50 | Pending | Pending | Pending | Depth experiment; implement only after earlier comparisons |

### Proposed follow-up experiments

Keep the 64x64 data, train/evaluation split, seed, latent size, optimizer settings, 50 epochs, and metric pipeline fixed unless a trial names that setting as its single change. Start each run from a fresh initialization with seed 42; do not fine-tune one trial's weights into another. Select the parent configuration for Trials 5-8 after comparing held-out FID/IS and fixed-seed sample grids, rather than automatically stacking every change.

1. **Trial 4 — update ratio.** Add a `generator_updates_per_discriminator` config setting, keep it at 1 for old trials, and set it to 2 for Trial 4. This gives G two optimizer steps for each D step. Keep all Trial 3 settings otherwise, so the update ratio is the only change. Trials 2-3 show low D loss and rising G loss; this is a targeted hypothesis, and the scores and sample grid decide whether it helps. Do not use the common WGAN-GP five-critic-step ratio with the current BCE DCGAN, because that belongs to a different objective and could make this discriminator stronger still.
2. **Trial 5 — discriminator spectral normalization (completed).** Applied spectral normalization to D's convolution layers only, keeping the Trial 2 settings otherwise unchanged. It did not improve IS/FID over Trials 1-2; see the Trial 5 results above and archived artifacts in `results/trial5_sn_d/`. See Miyato et al., [Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957).
3. **Trial 6 — generator width (completed).** Increased only `generator_channels` from 64 to 96, keeping D at 64 and otherwise matching Trial 2. IS/FID were 2.937/0.04191 versus Trial 2's 2.951/0.04236, a negligible metric difference without repeated seeds; the wider G did not show a clear gain.
4. **Trial 7 — generator activation.** Replace hidden G ReLU activations with LeakyReLU(0.2), keeping D's LeakyReLU and all widths fixed. Keep the final `tanh` output and image normalization unchanged. DCGAN's original architecture uses ReLU in G and LeakyReLU in D, so treat this as an empirical comparison rather than an assumed improvement; see [Radford et al.](https://arxiv.org/abs/1511.06434).
5. **Trial 8 — depth (optional).** Add one residual block at the 16x16 feature-map stage in both G and D, keeping widths fixed. This tests added depth as one matched architecture change. It will require a separate architecture version in checkpoints so evaluation reconstructs the correct model.

**Optional separate variant:** If the BCE DCGAN remains unstable after these controlled tests, consider a separate WGAN-GP implementation with critic loss, gradient penalty (start with lambda=10), no batch normalization in the critic, and its own documented update schedule. Do not compare it as a one-factor DCGAN tweak because it changes the training objective and critic constraints. The method is from Gulrajani et al., [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028).

**Observed resource cost:** Trial 3 took about 47 minutes total (23.3 minutes training, about 24 minutes for evaluation). Trial 4's two G updates per D update is slower; its first 31 epochs took about 23 minutes, so a complete run plus evaluation is likely around 55-65 minutes. At the latest check it had 7 training epochs and evaluation left, estimated around 25-30 minutes from that point.

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
- Trial 3 progress check (2026-11-16 about 20:02 PST): 28/50 epochs; epoch 28 D=0.1861, G=5.1025. Epoch 10 and 20 checkpoints saved; output directory was about 224 MB shortly before this check.
- Trial 3 final training result: 50/50 epochs in 23.3 minutes; final losses D=0.0570, G=7.6003. Its full checkpoint set and generated 64-image grid were saved.
- Trial 3 evaluation: Ignite 0.5.5, 4,073 held-out real images and 4,073 generated images at 299x299; IS=2.117535, FID=0.116542. Compared with Trial 2 (IS=2.951, FID=0.0424), both scores worsened. The very low D loss and high G loss suggest discriminator dominance; inspect the fixed-seed grid to distinguish poor quality from visual collapse.
- Storage cleanup after completed Trials 2-3: first removed each run's redundant `checkpoints/latest.pt` copy after confirming `epoch_050.pt` exists. When Trial 4 hit the account quota, removed intermediate epoch-10/20/30/40 checkpoints from Trials 2-3, retaining each final epoch-50 checkpoint, all grids, logs, and metric JSON. Also removed Trial 4's epoch-10/20/30 copies while retaining `latest.pt` at epoch 30 for resuming. Total additional space recovered: about 803 MB.
- Trial 4 implementation is prepared and smoke-checked locally: `train.py` supports `generator_updates_per_discriminator` (default 1), and `config_trial4.json` / `run_trial4_batch.sh` set it to 2 with all other settings matching Trial 3. Training logs report mean G loss across its two updates for fair per-batch comparisons.
- Trial 4 medium-GPU run launched 2026-11-16 about 20:38 PST as pod `xix015-2792513` on `its-dsmlp-n54.ucsd.edu`. The RTX PRO 6000 Blackwell MIG passed CUDA preflight. Initial progress: epoch 1 D=2.1033, mean G=0.8658; the 2:1 G:D setting was active.
- Trial 4 first attempt: pod failed around epoch 31 with `OSError: [Errno 122] Disk quota exceeded` while flushing loss logs. The last resumable checkpoint is `latest.pt` at epoch 30; no Trial 4 metrics are available yet.
- After the 803 MB GAN-checkpoint cleanup, `run_trial4_batch.sh` now resumes from `latest.pt` when present, preserving optimizer/RNG state and appending losses. The loss CSV ends at epoch 30, matching the resume point.
- Trial 4 resumed 2026-11-16 about 21:12 PST in pod `xix015-2829602` on a medium-class RTX A5000 node. CUDA preflight passed and it loaded `latest.pt` at epoch 30. Epoch 40 saved successfully; the run completed without further quota errors.
- Trial 4 final training result: 50/50 epochs; final losses D=0.1451, G=6.1784. The resumed 20 epochs took 13.2 minutes; total run elapsed time, including the failed attempt and evaluation, was about 49 minutes.
- Trial 4 evaluation: Ignite 0.5.5, 4,073 real and 4,073 generated images at 299x299; IS=2.807405, FID=0.163122. Relative to Trial 3, IS improved (2.118 to 2.807) but FID worsened (0.1165 to 0.1631). The generated grid has repeated visual motifs, so the 2:1 G:D update ratio did not resolve repetition. Trial 2 remains the best overall score pair (IS=2.951, FID=0.0424); Trial 1 still has the lowest FID (0.0372).
- Trial 2-4 report evidence (resolved configs, per-step losses, loss plots, evaluation JSONs, and sample grids) is archived in `results/` and pushed to GitHub. Removed the backed-up Trial 3 and Trial 4 output folders from DSMLP; retained Trial 2's epoch-50 checkpoint as the current model candidate. The outputs directory was about 91 MB before Trial 5.
- Trial 5 launch: pod `xix015-2870973` on a medium-class RTX A5000; batch size 64, D learning rate 5e-5, one G update per D update, and spectral normalization on all D convolution layers. It differs from Trial 2 only by D spectral normalization. PyTorch 2.9.1+cu128 passed CUDA preflight; output directory was about 2.7 MB at startup.
- Trial 5 final training result: 50/50 epochs completed in about 17.6 minutes on a medium-class RTX A5000; final losses D=0.0713, G=5.4023.
- Trial 5 evaluation: Ignite 0.5.5, 4,073 held-out real and 4,073 generated images at 299x299; IS=2.294966, FID=0.098651. Compared with Trial 2 (IS=2.951, FID=0.0424), spectral normalization worsened both metrics; it improved over Trial 3 but did not beat Trials 1-2.
- Trial 5 evidence is archived in `results/trial5_sn_d/` and pushed to GitHub. Removed its remote output folder after verifying the archive, recovering about 446 MB; deleted its completed pod. Retain Trial 2's epoch-50 checkpoint as the current model candidate; GAN outputs now use about 91 MB.
- Trial 6 final training result: 50/50 epochs in 16.8 minutes on a medium-class RTX A5000; final losses D=0.3699, G=3.5548. Checkpoints were saved at epochs 25 and 50.
- Trial 6 evaluation: Ignite 0.5.5, 4,073 held-out real and generated images at 299×299; IS=2.936973, FID=0.041910. Relative to Trial 2, IS decreased by 0.0138 and FID improved by 0.00045; these single-seed differences are too small to establish a reliable improvement.
- Trial 6 evidence is archived in `results/trial6_g_width96/`.
- Trial 7 stop: the batch script automatically began the activation run after Trial 6 evaluation, reaching epoch 4. The pod was deleted after the user's instruction to stop; no complete run, checkpoint, or metric result was produced. Do not count this partial attempt as a completed trial.
- Current status: Trials 2-6 complete; Trial 7 stopped before completion. The generator-width comparison did not materially improve Trial 2. Resume the activation/depth ablations only if time permits and explicitly requested.

## 6. Generate final samples and evaluate

- [x] Generate a fixed number of images from the selected checkpoint; save them under a dedicated output folder.
- [x] Create a grid of representative samples for the report, using a fixed random seed and no cherry-picking beyond clearly documenting the selection.
- [x] Compute IS on generated images.
- [x] Compute FID between generated images and the held-out real paintings (not the training split).
- [x] Keep the metric implementation, preprocessing, image count, and resize policy identical across runs. Record Ignite/PyTorch versions.
- [x] Report the number of real and generated images used. For Inception-based metrics, follow the selected implementation's input-size and value-range requirements.

### Evaluation results

- Selected report example/checkpoint: Trial 2 `outputs/trial2_lr_d_5e5/checkpoints/epoch_050.pt` for its highest IS (2.951) and near-best FID (0.0424); Trial 1 remains the lowest-FID run (0.0372).
- Metric implementation and versions: PyTorch-Ignite 0.5.5; PyTorch 2.9.1+cu128; torchvision 0.24.1+cu128.
- Real evaluation images: 4,073 held-out images.
- Generated images: 4,073 for metrics; 64 for the saved visual sample set.
- Resize / normalization details: model output/real images converted to RGB `[0,1]`, resized to 299×299 for Ignite Inception metrics.
- Inception Score (mean ± std, if available): Trial 2: 2.9507 (Ignite returned a scalar); Trial 1: 2.684.
- FID: Trial 2: 0.04236; Trial 1: 0.0372 (lowest among tested runs).
- Qualitative observations: Trial 1 and Trial 2 show painterly color fields and brush-like textures; both remain blurry, and repeated motifs occur.
- Limitations / caveats: Ignite's default FID implementation differs from `pytorch-fid`; do not compare the numeric value directly with scores from other FID implementations. The 64×64 input resolution limits fine detail.

## 7. Write the two-page report

- [x] Use the NeurIPS 2024 format and stay within two pages, including references.
- [x] Include data, model/training setup, results, and limitations/conclusion.
- [x] Include a compact hyperparameter table, generator/discriminator loss plot, generated-image grids, and IS/FID results.
- [x] Caption figures with run and sample count; state the metric protocol in the text.
- [x] Cite the WikiArt dataset and the metric implementation/tutorial.
- [x] Verify reported values against archived evaluations and loss logs.

### Report findings and results

- Main result: among seven complete configurations (baseline plus Trials 1-6), Trial 2 has highest IS (2.9507); Trial 1 has lowest FID (0.0372). Trial 6 is effectively tied with Trial 2 but does not establish a width benefit. Trial 2 is used as the balanced report example.
- Best configuration and why: Trial 2, batch 64, $\mathrm{lr}_G=2\times10^{-4}$, $\mathrm{lr}_D=5\times10^{-5}$, Adam $(0.5,0.999)$; it combines the highest IS with near-best FID.
- What the generated images capture well: painterly color fields and coarse textured brush-like patterns.
- Common artifacts or failure modes: blur, incoherent forms, and repeated motifs at 64×64.
- What the scores suggest (and do not establish): the lower D learning rate improved the baseline; IS/FID rank these runs under one ImageNet-based protocol but do not establish human-rated artwork quality.
- Limitations: single seed per configuration, low resolution, ImageNet-domain metric bias, and a compute-limited search.
- Follow-up experiments: repeated seeds, higher resolution, and the deferred G activation/depth ablations.
- Report source/PDF: `report/report.tex` and `report/report.pdf`.

## Final checklist

- [x] Training script/config and reproducibility details are saved.
- [x] Loss curves and final sample grids are saved.
- [x] IS/FID scores and evaluation protocol are recorded.
- [x] Final checkpoint candidate is identified (Trial 2 epoch 50; checkpoint retained on DSMLP).
- [x] Two-page report is complete and figures are readable.
