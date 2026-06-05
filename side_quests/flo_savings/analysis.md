# FLO performance analysis

**Period:** November 1, 2025 – June 5, 2026

This note compares electricity spending under the Forward Looking Optimizer (FLO) against a baseline that represents what a homeowner would have paid without thermal storage or price-aware control. All cost totals below are summed over **hours when FLO was running**.

---

## Overview

Each hourly record includes heat pump electricity use, distribution energy, weather, and locational marginal price (LMP). We compute two costs for the FLO-active period:

| Metric | What it represents |
| --- | --- |
| **FLO cost** | Actual electricity bought to run the heat pump, priced at hourly LMP |
| **Baseline cost** | Estimated electricity that would have been needed to meet the same thermal load immediately, without storage or smart scheduling |

The timeline plots show when FLO was active (green) versus rule-based control (gray). The title reports both cost totals for that house.

---

## Methodology

### FLO electricity cost

For each hour $h$ when FLO is running:

$$
C^{\text{FLO}}_h = E^{\text{HP, el}}_h \cdot \frac{\text{LMP}_h}{1000}
$$

where $E^{\text{HP, el}}_h$ is heat pump electricity use (kWh) and $\text{LMP}_h$ is the locational marginal price (USD/MWh). The factor of 1000 converts MWh pricing to kWh consumption.

The reported FLO total is $\sum_h C^{\text{FLO}}_h$ over FLO-active hours.

### Baseline electricity cost

The baseline estimates how much electricity the heat pump would have consumed in each hour if it had delivered the thermal load on demand, with no storage buffer and no price optimization. This requires hourly estimates of **load** and **COP**.

$$
C^{\text{base}}_h = \frac{L_h}{\text{COP}_h} \cdot \frac{\text{LMP}_h}{1000}
$$

The reported baseline total is $\sum_h C^{\text{base}}_h$ over the same FLO-active hours.

#### COP model

COP is modeled as a linear function of outdoor air temperature (OAT), with a fixed value at very cold temperatures. Parameters are set per house.

$$
\text{COP}_h =
\begin{cases}
\text{COP}_{\min} & \text{if } \text{OAT}_h < T_{\min} \\
\beta_0 + \beta_{\text{oat}} \cdot \text{OAT}_h & \text{otherwise}
\end{cases}
$$

| Symbol | Meaning |
| --- | --- |
| $\beta_0$ | COP intercept (`cop_intercept`) |
| $\beta_{\text{oat}}$ | OAT coefficient (`cop_oat_coeff`) |
| $\text{OAT}_h$ | Outdoor air temperature at hour $h$ (°F) |
| $\text{COP}_{\min}$ | COP used below the cold-temperature threshold (`cop_min`) |
| $T_{\min}$ | OAT threshold (°F) below which COP is held constant (`cop_min_oat_f`) |

#### Load model

Thermal load is estimated in two steps.

**Step 1 — Predict distribution energy from weather**

A linear regression is fit on all hours in the dataset (not only FLO-active hours) to predict hourly distribution energy from OAT and wind speed:

$$
\hat{D}_h = \alpha + \beta \, \text{OAT}_h + \gamma \, W_h \left(65 - \text{OAT}_h\right)
$$

where $W_h$ is wind speed (mph) and $\alpha$, $\beta$, and $\gamma$ are fitted from the data.

**Step 2 — Scale to total heat pump output**

The predicted distribution curve is scaled so its total matches the ratio between aggregate heat pump thermal output and aggregate measured distribution energy:

$$
L_h = \hat{D}_h \cdot \frac{\sum_h E^{\text{HP, th}}_h}{\sum_h D_h}
$$

where $E^{\text{HP, th}}_h$ is measured heat pump thermal output and $D_h$ is measured distribution energy. This scaling maps distribution-side load to the thermal output the heat pump would need to deliver in a no-storage scenario.

---

## Results

### Beech

![Beech FLO timeline](results/beech_flo_timeline.png)

### Elm

![Elm FLO timeline](results/elm_flo_timeline.png)

### Fir

![Fir FLO timeline](results/fir_flo_timeline.png)

### Maple

![Maple FLO timeline](results/maple_flo_timeline.png)

### Oak

![Oak FLO timeline](results/oak_flo_timeline.png)
