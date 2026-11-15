# ECE 285 HW2 Report Outline (two pages, NeurIPS format)

Replace each bracketed prompt with measured details. Keep the final paper within the instructor's page-limit interpretation.

## Title and abstract

**Title:** [Unconditional DCGAN for WikiArt Paintings]

**Abstract (3–5 sentences):** State the task, dataset scale, model and resolution, training setup, key IS/FID results, and one qualitative observation.

## 1. Introduction and data

We train an unconditional generative adversarial network to synthesize painting-like images from the WikiArt dataset [dataset citation]. The processed dataset contains [valid image count] images across [style-folder count] style folders. We combine the folders into one training set and do not condition generation on style labels. We reserve [count or percentage] images for held-out evaluation.

## 2. Method

Describe the DCGAN generator and discriminator in 2–4 sentences: [latent dimension], [convolutional up/downsampling stages], [normalization and activations], and [loss]. State the image preprocessing policy and output resolution.

| Setting | Value |
|---|---:|
| Resolution | [ ] |
| Batch size | [ ] |
| Latent dimension | [ ] |
| Epochs | [ ] |
| Learning rates (G / D) | [ ] |
| Adam betas | [ ] |
| Training hardware | [ ] |

## 3. Experiments and results

Report the training split, seed, checkpoint-selection rule, and metric implementation/sample counts. Explain that lower FID is better and that IS evaluates generated-image class confidence/diversity but does not compare directly with real samples.

**Results:** IS = [mean ± std]; FID = [value] using [number] generated and [number] held-out real images.

**Figure 1:** Generator and discriminator losses over [steps/epochs].

**Figure 2:** Grid of [number] generated images from checkpoint [epoch/run].

Discuss visible strengths, artifacts, diversity/mode collapse, and how these observations relate to IS/FID: [findings].

## 4. Conclusion and limitations

Summarize the best observed outcome and one limitation, such as low resolution, compute budget, style imbalance, or the limits of ImageNet-based metrics for artwork. State the most useful next experiment.

## References

- WikiArt dataset: [Steubk, “WikiArt,” Kaggle, URL and access date].
- Metric implementation/tutorial: [PyTorch-Ignite FID/IS reference].
- GAN architecture: [DCGAN reference required by course/report style].
