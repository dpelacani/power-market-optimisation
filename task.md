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

| Generator | Type       | Capacity (MW) | Marginal Cost (£/MWh) | Startup Cost (£) | Min Run Time (hrs) | Min Stable Generation (MW) |
|-----------|------------|---------------|------------------------|------------------|--------------------|--------------------|
| G1        | Nuclear    | 400           | 10                     | 50,000           | 8                  | 150                |
| G2        | Coal       | 300           | 25                     | 8,000            | 4                  | 100                |
| G3        | Gas (CCGT) | 200           | 45                     | 2,000            | 2                  | 60                 |
| G4        | Gas (OCGT) | 100           | 80                     | 500              | 1                  | 0                  |
| G5        | Oil Peaker | 50            | 120                    | 200              | 1                  | 0                  |

**Notes:**
- All generators are offline at t=0 (start of the day)
- Capacity = maximum amount of power a generator can produce
- Marginal cost = cost per MWh of electricity produced while running
- Startup cost = one-time cost incurred each time a generator is switched on
- Min run time = once switched on, a generator cannot be switched off before this many hours
- Min Stable Generation = minimum amount of power a generation that is on must produce in order to be stable

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

Here is the addition:

---

### Part 4 — Lagrange Multipliers, Dual Variables and Shadow Prices

In unconstrained optimisation you minimise a function $f(x)$ freely. In constrained optimisation you minimise $f(x)$ subject to $g(x) = b$ — and Lagrange multipliers are the tool that bridges the two.

The core idea: attach a multiplier $\lambda$ to each constraint and fold it into the objective, forming the **Lagrangian**:

$$\mathcal{L}(x, \lambda) = f(x) - \lambda \cdot (g(x) - b)$$

At the optimum, $\lambda$ tells you exactly how sensitive the optimal objective value is to a marginal relaxation of the constraint:

$$\lambda = \frac{\partial f^*}{\partial b}$$

In plain terms: **if you loosen constraint $g$ by one unit, the optimal cost changes by $\lambda$**. This is why $\lambda$ is called a **shadow price** — it is the implicit price the system places on each constraint being binding.

**Why this matters in power markets**

Every constraint in the UC problem has a shadow price with a direct economic interpretation:

| Constraint | Shadow Price Meaning |
|---|---|
| Demand balance at hour $t$ | Cost of serving 1 extra MW at hour $t$ — i.e. the **SMP** |
| Capacity of generator $g$ at hour $t$ | Value of 1 extra MW of capacity — i.e. the **capacity price** |
| Minimum stable generation | Cost savings from lowering the operating floor by 1 MW |
| Minimum run time | Cost savings from freeing a generator to switch off 1 hour earlier |

The demand balance multiplier is the most important: it is precisely the System Marginal Price you computed in Parts 1 and 2, now derived formally from duality theory rather than from reading off the last dispatched generator's cost.

**The MILP complication**

Shadow prices are well-defined for continuous linear programmes (LPs). The UC problem is a MILP — its binary variables make it non-convex, and strict duality theory does not apply directly. The standard practical approach is to **fix the binary variables** at their optimal values from the MILP solution and re-solve the resulting LP. This LP has the same optimal dispatch but now admits well-defined dual variables, which are meaningful as local price signals around the optimal solution.

---

#### Tasks

**4.1 — Extract shadow prices from the LP relaxation**

Take the optimal binary solution $u^*$ and $v^*$ from Part 2. Fix these values and re-solve as a pure LP by replacing all binary variables with fixed constants. Extract the dual values representing shadow prices.

**Question**: *How do these shadow prices compare to the SMP values you computed in Part 2? Are they identical? If not, why might they differ?*

```python
# Fix binary variables at their optimal values and re-solve as LP
lp_problem = pulp.LpProblem("UC_LP_Relaxation", pulp.LpMinimize)

# Re-declare p as the only free variable
# Add all constraints with u and v fixed at their MILP-optimal values
# Solve and extract dual variables via constraint.pi
```

For each hour $t$, extract the dual variable on the demand balance constraint:

```python
for t in range(T):
    lam = lp_problem.constraints[f"Demand_Balance_Hour_{t}"].pi
    print(f"Hour {t:02d}: shadow price = £{lam:.2f}/MWh")
```


---

**4.2 — Capacity shadow prices**

Extract the dual variable on the capacity constraint for each generator at each hour:

```python
for i in range(Ng):
    for t in range(T):
        mu = lp_problem.constraints[f"Capacity_Generator_{i}_Hour_{t}"].pi
```

Plot a heatmap of capacity shadow prices (generators × hours).

**Questions to answer:**
- Which generators have non-zero capacity shadow prices, and at which hours?

- What are the two conditions required for a capacity shadow price to be non-zero — and why is binding alone not sufficient?

- At the hours of highest demand, which generator's capacity is most valuable? What would it be worth to add 10 MW (assuming linearity) to that generator?

---

**4.3 — Interpreting the minimum stable generation constraint**

Extract the dual variable on the minimum stable generation constraint for Coal (G2) during the midday dip hours (13-14):

**Questions to answer:**
- Is this constraint binding during the dip? What does that tell you about Coal's dispatch at those hours?

- What does the shadow price tell you about the cost of the minimum stable generation requirement? In other words, how much would the system save if Coal could operate at zero output rather than 100 MW while remaining committed?

- Connect this back to the UC saving from Part 2: the optimal solution kept Coal on through the dip at minimum stable output. The shadow price here quantifies exactly what that decision cost per MW of floor output.

---

**Outputs**

- Shadow price time series plot for the demand balance constraint (should closely match the SMP from Part 2)
- Heatmap of capacity shadow prices across generators and hours
- Written answers to the questions in 4.2 and 4.3


---

## Deliverables

- A single well-structured Python script or notebook
- All plots clearly labelled with titles, axes, and legends
- Brief written answers to the three questions (as comments or markdown cells)

## Libraries

You may use: `numpy`, `pandas`, `matplotlib`, `PuLP`

Install PuLP if needed: `pip install pulp`
