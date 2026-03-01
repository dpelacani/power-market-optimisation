# Economic Dispatch & Unit Commitment in a Simple Power Market

## Background

You are an energy modeller at a consultancy. A grid operator needs to schedule a fleet 
of generators over a 24-hour period to meet hourly electricity demand at minimum cost.

This is one of the most fundamental problems in power systems: the **Unit Commitment 
and Economic Dispatch** problem. It combines two decisions:

- **Unit Commitment (UC)**: *Which* generators should be switched on each hour? 
  (binary on/off decision)
- **Economic Dispatch (ED)**: Given which generators are on, *how much* should each 
  produce? (continuous allocation decision)

---

## The Generator Fleet

You have five generators with the following characteristics:

| Generator | Type       | Capacity (MW) | Marginal Cost (£/MWh) | Startup Cost (£) | Min Run Time (hrs) |
|-----------|------------|---------------|------------------------|------------------|--------------------|
| G1        | Nuclear    | 400           | 10                     | 50,000           | 8                  |
| G2        | Coal       | 300           | 25                     | 8,000            | 4                  |
| G3        | Gas (CCGT) | 200           | 45                     | 2,000            | 2                  |
| G4        | Gas (OCGT) | 100           | 80                     | 500              | 1                  |
| G5        | Oil Peaker | 50            | 120                    | 200              | 1                  |

**Notes:**
- All generators are offline at t=0 (start of the day)
- Marginal cost = cost per MWh of electricity produced while running
- Startup cost = one-time cost incurred each time a generator is switched on
- Min run time = once switched on, a generator cannot be switched off before this many hours

---

## Demand Profile

Hourly demand (MW) over 24 hours:

```python
demand = [
    280, 265, 250, 245, 245, 260,   # 00:00 - 05:00 (night trough)
    310, 420, 480, 510, 530, 545,   # 06:00 - 11:00 (morning ramp)
    540, 530, 380, 375, 510, 555,   # 12:00 - 17:00 (midday plateau)
    610, 625, 590, 545, 460, 365,   # 18:00 - 23:00 (evening peak then falloff)
]
```

---

## Tasks

### Part 1 — Merit Order Dispatch (Greedy Baseline)

Implement a **merit order dispatcher** that, for each hour:
1. Ignores startup costs and minimum run times entirely
2. Ranks generators by marginal cost (cheapest first)
3. Dispatches generators in that order until demand is met
4. Sets the **market clearing price** as the marginal cost of the last (most expensive) 
   generator needed to meet demand — this is the **System Marginal Price (SMP)**

**Output:**
- For each hour: which generators run, how much each produces, and the SMP
- Total production cost across the day (marginal costs only, no startup costs)
- A stacked bar chart of hourly generation mix by generator type
- A line plot of the System Marginal Price over 24 hours

**Question to answer:** What is the total generation cost under the greedy merit order?

---

### Part 2 — Unit Commitment with Linear Programming

Now implement the **full Unit Commitment problem** as a Mixed-Integer Linear Programme 
(MILP) using `PuLP`.

**Decision variables:**
- `u[g, t]` ∈ {0, 1} — whether generator `g` is online at hour `t`
- `p[g, t]` ≥ 0 — power output (MW) of generator `g` at hour `t`
- `v[g, t]` ∈ {0, 1} — whether generator `g` starts up at hour `t` (i.e., was off at t-1, on at t)

**Objective:** Minimise total cost:
$$\text{minimise} \sum_{g,t} \left( c_g \cdot p_{g,t} + SC_g \cdot v_{g,t} \right)$$

where $c_g$ is marginal cost and $SC_g$ is startup cost.

**Constraints:**
1. **Demand balance**: total generation must meet demand each hour
   $$\sum_g p_{g,t} = D_t \quad \forall t$$

2. **Capacity limits**: a generator can only produce between 0 and its capacity, and only 
   if it is online
   $$0 \leq p_{g,t} \leq P_g^{max} \cdot u_{g,t} \quad \forall g, t$$

3. **Startup detection**: a startup occurs when a unit goes from off to on
   $$v_{g,t} \geq u_{g,t} - u_{g,t-1} \quad \forall g, t > 0$$

4. **Minimum run time**: once started, a generator must run for at least `MRT_g` hours
   $$\sum_{\tau=t}^{\min(t + MRT_g - 1, T)} u_{g,\tau} \geq MRT_g \cdot v_{g,t} \quad \forall g, t$$

5. **Minimum stable generation**: a committed generator must produce at least its minimum stable output
$$p_{g,t} \geq P_g^{min} \cdot u_{g,t} \quad \forall g, t$$

**Output:**
- Optimal schedule: for each hour, which generators are committed and at what output
- Total optimal cost (including startup costs)
- Same stacked bar chart and SMP line plot as Part 1, now reflecting optimal dispatch
- The **commitment schedule** as a heatmap (generators × hours, shaded by output)

**Question to answer:** What is the total cost saving of the optimal UC solution versus 
the greedy merit order (accounting for startup costs in both)?

---

### Part 3 — Analysis and Interpretation

1. **Price duration curve**: plot the System Marginal Price sorted from highest to lowest 
   for both the greedy and optimal solutions. What does the shape tell you?

2. **Startup cost impact**: run the MILP with startup costs set to zero. How does the 
   schedule change? What does this reveal about the role of commitment costs?

3. **Scarcity hour**: identify the hour(s) where the system is under the most stress 
   (demand closest to total available capacity). What is the SMP at those hours and why?

---

## Deliverables

- A single well-structured Python script or notebook
- All plots clearly labelled with titles, axes, and legends
- Brief written answers to the three questions (as comments or markdown cells)

## Libraries

You may use: `numpy`, `pandas`, `matplotlib`, `PuLP`

Install PuLP if needed: `pip install pulp`
