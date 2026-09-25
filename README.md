# Contact Mode Diagnostics
This experiment investigates how the latent dynamics model behaves in changing contact regimes.
We separate that into several questions:
1. **E1:** Does prediction error concentrate near contact transitions?
2. **E1A** How much does contact-onset regularization affect prediction error?

## Controlled simulation environment
We deliberately used a minimal system to make the mechanisms and labels inspectable:
- A 1kg sphere with two translational DOF
- $x$: normal to a fixed wall; $y$: tangential
- No rotation, gravity, table, or robot
- Friction coefficient $\mu$=0.5
- Force motors acting along both coordinates
- MuJoCo 3.12.0, Newton solver, elliptic friction cone, solver tolerance $10^{-10}$
- Primary simulation and prediction interval: 0.125 ms

| Protocol | Purpose | Excitation |
|---|---|---|
| A2 | Quasi-stick→slide→quasi-stick under continuous normal contact | Constant normal preload; cyclic tangential force |
| A3 | Repeated controlled separation/recontact | Moving support connected through a spring–damper force |
| A2 quasi-stick control | Force variation without clear sliding | Tangential force kept below the intended sliding regime |
| A3 contact control | Normal dynamics without separation | Small support oscillations |
| A3 free control | Dynamics without wall contact | Support motion sufficiently far from the wall |

## E1 Experimental Design
A2 normal preloads over 8-12 N and varied the tangential force schedule. It aimed to produce repeated transitions while returning to stable quasi-stick.
A3 applied:
$$
F_x = 1000(x_{support} - q_x) + 30(v_{support} - v_x), F_y = 0
$$
$$
x_{support}(t) = x_0 + Asin(2\pi ft + \phi), v_{support}(t)=2\pi fAcos(2\pi ft + \phi)
$$
For transition episodes, support amplitudes were 15-25 mm and frequencies 1.5-2.5 Hz. Initial tangential position, support phase, and nominal preload also varied. Episodes retained three analysis cycles.
Calibration was performed for labeling A2 so to reduce ambiguity on classification. The calibration procedures used:
- Fit preloads: 8, 10, 12 N
- Separate Validation preloads: 9, 11 N
- Six subthreshold force ratios per preload (total: 30)
- Five additional suprathreshold validation cases

The fitted maximum creep speed was approximately $4.898 \times 10^{-4}$ ms. By the acquired max. speed, we use the frozen primary thresholds as:

| Quantity | Frozen value |
|---|---|
| Quasi-stick speed ceiling | $6.122 \times 10^{-4}$ m/s |
| Clear-slide speed floor | $2.449 \times 10^{-3}$ m/s |
| Friction-utilization boundary | $\rho = 0.98$ |
| Loaded-contact force threshold | $F_n > 0.1$ N |

Thus, for loaded A2 contact:
- QUASI_STICK: low speed and $\rho < 0.98$
- SLIDE: speed above the slide floor and $\rho >= 0.98$
- BOUNDARY: remaining ambiguous cases

## Learned Dynamics Baseline
Both experiments used the same mode-agnostic MLP:
$$
(q_x, q_y, v_x, v_y, F_x, F_y) \to (\delta q_x, \delta q_y, \delta v_x, \delta v_y)
$$
Architecture uses three hidden layers of width 128 with SiLU activations. No mode labels, contact forces, event times, or support phase were supplied as additional inputs.

Training used:
- Adam, lr=$10^{-3}$
- Batch size: 1024
- Maximum 200 epochs
- Early stopping patience of 20 epochs
- Seed 42
- Training-set normalization, reused for validation
- Normalized MSE for optimization
- Episode-balanced validation MSE for checkpoint selection

## E1 Result
Across 5 validation episodes, sample-weighted component RMSE was:

| Component | RMSE |
|---|---|
| $q_x$ | 0.3412 µm |
| $q_y$ | 0.003207 µm |
| $v_x$ | 0.03465 mm/s |
| $v_y$ | 0.0002714 mm/s |

<p align="center">
  <img src="E1/analysis/20260923_160152_584385/a3_0004_recontact.png" width="51.1%" alt="a3_0004 recontact windows">
  <img src="E1/analysis/20260923_160152_584385/a3_0004_recontact_zoom.png" width="46.9%" alt="a3_0004 recontact zoom">
</p>

We observe $\pm 5$ ms recontact window contained 99.13% of the squared $v_x$ prediction error while covering only 2.10% of eligible samples. We also observe mode-dependent error structure for A2, however, slide $\to$ stick showed higher prediction error compared to stick $\to$ slide.
Therfore, E1 can be concluded that the prediction error can be strongly localized around recontact in this controlled simulator and model.

## E1A Experimental Design
We employ two conditions:
| Condition | Contact `solimp` |
|---|---|
| Original | `[0.9, 0.95, 0.001, 0.5, 2]` |
| Modified | `[0, 0.95, 0.001, 0.5, 2]` |

Mass, support stiffness/damping, contact `solref`, primary timestep, and learning architecture were held fixed. Moreover, each pair shared its initial conditions, support-drive parameters, and splits. Actual force histories could differ, because the spring-damper controller depends on each trajectory's evolving state.

E1A contains only A3 families:
| Family | Pairs |
|---|---|
| Separation/recontact | 20 |
| Sustained-contact control | 20 |
| Free-motion control | 20 |

## E1A Result
Recontact was identified geometrically, and each condition's error was aligned to its own time event:
$$
\tau = t_i - t_{recontact}
$$
We used primary window as $\pm 5$ ms, with sensitivity width $\pm 2, \pm 10, \pm 20$. We compute RMSE directly from the original sample to check whether sharp behavior persists upon contact.

Validation results were:
| A3 episode | Original ±5 ms RMSE | Modified ±5 ms RMSE | Reduction |
|---|---|---|---|
| a3_0006 | 0.5467 mm/s | 0.2090 mm/s | 61.8% |
| a3_0007 | 0.5796 mm/s | 0.2566 mm/s | 55.7% |
| a3_0017 | 0.6834 mm/s | 0.0932 mm/s | 86.4% |

<p align="center">
  <img src="E1A/analysis/figures/recontact_a3_0006_cycle0.png" width="80%" alt="a3_0006 recontact cycle 0, original vs modified">
</p>
Increasing the constraint regularization successfully eliminates the error spike at recontact, and this benefit persists over a wider window. However, this creates a trade-off: prediction accuracy deteriorates during other parts of the cyclic contact trajectory. This occurs because the added regularization shifts the overall learned error distribution.

To isolate this effect, we analyzed the error across three distinct regions. Each region was defined dynamically based on the specific condition's trajectory:

| Trajectory Region | Selection Rule |
| :--- | :--- |
| **Near recontact** | Within $\pm 5$ ms of a geometric recontact |
| **Shallow contact** | Gap $\le 0$, penetration 0–1 mm *(Must be >20 ms away from any separation or recontact)* |
| **Free flight** | Gap $> 0$ *(Must be >20 ms away from any separation or recontact)* |

### Performance Breakdown
The table below shows the episode-balanced $v_x$ RMSE evaluated over the three A3 validation episodes. 
*(Calculated as: $\mathrm{RMSE}_{R} = \sqrt{\tfrac{1}{3}\sum_j \mathrm{RMSE}_{j,R}^2}$)*

| Region | Original (mm/s) | Modified (mm/s) | Change* |
| :--- | ---: | ---: | :--- |
| Near recontact | 0.6061 | 0.1985 | **−67.2%** (Better) |
| Shallow contact | 0.01673 | 0.03321 | **+98.5%** (Worse) |
| Free flight | 0.006910 | 0.003749 | **−45.7%** (Better) |

*\*Change $= 100 \times (\mathrm{RMSE}_{\text{modified}} / \mathrm{RMSE}_{\text{original}} - 1)$. Negative values indicate an improvement.*