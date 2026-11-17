# GAN result artifacts

These compact files preserve the report evidence for completed runs. The large `.pt` model checkpoints remain outside GitHub.

Each trial directory includes its resolved run configuration, per-step loss CSV, loss curve, Ignite evaluation JSON, a generated sample grid, and the fixed-noise epoch-50 training grid. T2 and T3 use the same 4,073-image held-out evaluation split and Ignite 0.5.5 protocol documented in `plan.md`.
