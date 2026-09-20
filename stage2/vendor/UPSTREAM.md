# Upstream training scaffold

Target upstream: <https://github.com/wouterkool/attention-learn-to-route>

It is MIT licensed and supplies the attention encoder, masked pointer decoder,
REINFORCE rollout baseline, dynamic TSP generation, training and checkpoint
patterns used as the reference for this reproduction.

The initial clone was blocked by the current network. Before a training run,
fetch a pinned snapshot with:

```powershell
git clone https://github.com/wouterkool/attention-learn-to-route.git stage2/vendor/attention-learn-to-route
git -C stage2/vendor/attention-learn-to-route rev-parse HEAD
```

Record the resulting SHA and retain the upstream `LICENSE`. The Stage 2 modules
implement the required wind ATSP and energy-feasibility extensions; upstream's
symmetric Euclidean TSP cost must not be used for this project.
