# Arm features (agent/)

- arms: **1,899** active (index frozen, append-only)
- features: **288** dimensions
- spec version: `arm-spec-v1` | projection: `svd64-v1`
- row-order sha256: `424efdf64b6c1d04...`

## Feature blocks

| block | dims |
|---|---|
| difficulty (one-hot + imputed flag) | 4 |
| duration (effort, calendar, indicators) | 4 |
| popularity (shrunk rating, counts, indicators) | 5 |
| access / pace / platform | 7 |
| family one-hot | 15 |
| skills multi-hot | 188 |
| text embedding (frozen SVD) | 64 |

## Firewall

This stage reads only `data/catalog/`. It never touches
`data/world/` -- no prerequisite edges, no study plans. Asserted in
code at the end of `run()`, not left as a comment. Prerequisite
satisfaction must stay a latent property the bandit discovers from
features; handing it over directly while the reward credits it
would let the bandit win by construction.

## Dimensionality note

d = 288 over 1,899 arms, dominated by the
188-dim skill multi-hot. Linear-bandit regret scales
with d, so for the actual policy you will likely want interaction
features (user-skill cosine, level gap, prereq coverage) computed
from these columns rather than feeding all of them as context. The
raw multi-hot is kept because those interaction features need it.
