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

## E1 data generation and labeling
A2 normal preloads over 8-12 N and varied the tangential force schedule. It aimed to produce repeated transitions while returning to stable quasi-stick.
A3 applied:
$$
F_x = 1000(x_{support} - q_x) + 30(v_{support} - v_x), F_y = 0
$$
$$
x_{support}(t) = x_0 + Asin(2\pi ft + \phi), v_{support}(t)=2\pi fAcos(2\pi ft + \phi)
$$
For transition episodes, support amplitudes were 15-25 mm and frequencies 1.5-2.5 Hz. Initial tangential position, support phase, and nominal preload also varied. Episodes retained three analysis cycles.
