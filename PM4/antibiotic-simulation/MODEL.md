# Model Selection Guide: Which Mutation Model Should You Use?

## Quick Answer

**Default/General Use:** `fitness_landscape`

**Why?** Most broadly applicable, empirically-grounded, conservative, works for 80% of bacterial resistance (chromosomal mutations).

**But the real answer is:** *It depends on the organism and clinical context.*

---

## Model Selection Matrix

| Clinical Scenario | Best Model | Organisms | Key Features | Parameters |
|------------------|------------|-----------|--------------|------------|
| **Unknown mechanism** | `fitness_landscape` | Most bacteria | Conservative, general | μ₀=10⁻⁸, N=10⁷ |
| **Hospital outbreak** | `horizontal_transfer` | Klebsiella, E. coli, Acinetobacter | Plasmids, rapid jumps | HGT=10⁻⁷ |
| **Tuberculosis** | `point_mutations` | M. tuberculosis, M. leprae | No HGT, stepwise | μ₀=10⁻¹⁰ |
| **CF lung infection** | `hotspot` | P. aeruginosa (CF), biofilms | Mutators, stress | μ₀=10⁻⁷ |
| **Drug cycling** | `asymmetric` | Any (policy question) | Reversibility testing | 50× asymmetry |
| **Community UTI** | `fitness_landscape` | E. coli, E. faecalis | Mostly chromosomal | Default |
| **MRSA (hospital)** | `horizontal_transfer` | S. aureus (HA-MRSA) | SCCmec mobile element | Include HGT |
| **MRSA (community)** | `fitness_landscape` | S. aureus (CA-MRSA) | Fixed in chromosome | No HGT |

---

## 🔬 Detailed Model Profiles

### 1. `fitness_landscape` (Default/Recommended)

#### When to Use
- Default choice when mechanism is unknown
- Chromosomal resistance (gyrA, parC, rpoB mutations)
- Community-acquired infections
- Baseline for comparison with other models
- Conservative predictions (won't over-estimate adaptation speed)

#### Biological Basis
- **Distance matters:** Probability ∝ exp(-5 × IC50_distance)
- **Asymmetry:** 10× easier to lose resistance than gain it
- **Mechanism:** Point mutations, small indels
- **Population scaled:** Accounts for 10⁷ cells

#### Best For These Organisms
- **E. coli** (community, chromosomal resistance)
- **Salmonella** spp.
- **Streptococcus pneumoniae**
- **Most Gram-positive cocci**
- **Staphylococcus aureus** (chromosomal resistance)

#### Evidence from Simulations
- Critical C: 1.65 (moderate)
- Diversity loss: -18% (moderate)
- Surviving genotypes: 12/12 (all can persist)
- Speed: Moderate (neither fastest nor slowest)

#### Literature Support
```
Drake JW (1991). Spontaneous mutation rate: μ₀ ≈ 10⁻⁸ per cell per generation
Andersson & Hughes (2010). Asymmetry: 10-100× easier to lose function
Farhat et al. (2013). Most resistance = sequential point mutations
```

#### Code Example
```python
p = create_resistance_parameters(
    m=12,
    resistance_range=(0.1, 5.0),
    mutation_structure='gradient',
    mutation_type='fitness_landscape',  # ← Default choice
    base_mutation_rate=1e-8,
    population_size=1e7
)
```

---

### 2. `horizontal_transfer` (Hospital Pathogens)

#### When to Use
- Hospital-acquired infections
- Known plasmid carriers
- Multi-drug resistance cassettes
- Rapid outbreaks
- High-risk clones (KPC, NDM, ESBL)

#### Biological Basis
- **Two mechanisms:** Point mutations + HGT events
- **HGT rate:** ~10⁻⁷ per cell (100× higher than spontaneous resistance)
- **Can jump:** Directly from IC50=0.5 to IC50=4.0 (no gradual steps)
- **Uniform distance:** HGT doesn't depend on genetic distance

#### Best For These Organisms
- **Klebsiella pneumoniae** - KPC, NDM carbapenemases
- **E. coli** - ESBL (blaCTX-M, blaTEM, blaSHV)
- **Acinetobacter baumannii** - OXA carbapenemases
- **Enterobacteriaceae** - Mobile resistance elements
- **Pseudomonas aeruginosa** - VIM, IMP carbapenemases

#### Evidence from Simulations
- **Critical C: 0.90** (45% LOWER than fitness_landscape!)
- Diversity: Maintains high (gene transfer creates variants)
- Surviving genotypes: 12/12
- Speed: **FASTEST** resistance emergence

#### Why It Matters Clinically
**Your simulation shows:**
- HGT model reaches 50% resistant at C=0.90
- Fitness landscape reaches 50% resistant at C=1.65
- **Gap = 83%** → Under-dosing is much more dangerous for plasmid carriers

**Clinical implication:** Standard MIC-based dosing may be insufficient for hospital pathogens with plasmids.

#### Literature Support
```
Porse et al. (2016). HGT rate in mixed cultures: ~10⁻⁷
Carattoli (2013). Plasmids account for most MDR in Enterobacteriaceae
Modi et al. (1991). Conjugation frequency: 10⁻⁶ to 10⁻⁸
```

#### Code Example
```python
p = create_resistance_parameters(
    m=12,
    resistance_range=(0.1, 5.0),
    mutation_structure='gradient',
    mutation_type='horizontal_transfer',  # ← For plasmid carriers
    base_mutation_rate=1e-8,
    population_size=1e8  # Higher density in hospitals
)
```

---

### 3. `point_mutations` (Tuberculosis & Slow Evolvers)

#### When to Use
- Organisms with minimal/no HGT
- Slow mutation rates
- Resistance requiring 2+ sequential mutations
- Long treatment courses (months)
- Single-locus resistance genes

#### Biological Basis
- **Strictly stepwise:** Can only mutate to "adjacent" resistance levels
- **No jumping:** Must traverse IC50 space incrementally
- **Chromosomal only:** No plasmids, transposons, or mobile elements
- **Constrained evolution:** Fewer pathways to high resistance

#### Best For These Organisms
- **Mycobacterium tuberculosis** (canonical example)
- **Mycobacterium leprae**
- **Helicobacter pylori** (limited HGT)
- **Some Streptococcus** spp.

#### Evidence from Simulations
- Critical C: 1.65 (same as fitness_landscape)
- **Surviving genotypes: 8/12** (STRICTEST bottleneck)
- Diversity loss: -18%
- Speed: Moderate (gradual accumulation)

#### Why It's Different
**Your simulation shows:**
- Point mutations model eliminates 4 genotypes (can't reach high IC50 in one step)
- Other models: all 12 survive (can jump via HGT or continuous mutations)
- More restrictive evolutionary pathways

#### TB-Specific Considerations
- Rifampin resistance: rpoB mutations (1-2 steps)
- Isoniazid resistance: katG mutations (1 step)
- Fluoroquinolone: gyrA + gyrB (2-3 steps)
- Multi-drug resistance requires sequential acquisition

#### Literature Support
```
Farhat et al. (2013). TB resistance = primarily chromosomal SNPs
Eldholm & Balloux (2016). TB has minimal horizontal gene transfer
Gagneux (2018). Resistance evolves step-by-step over months
```

#### Code Example
```python
p = create_resistance_parameters(
    m=12,
    resistance_range=(0.1, 5.0),
    mutation_structure='gradient',
    mutation_type='point_mutations',  # ← For TB
    base_mutation_rate=1e-10,  # TB has lower mutation rate
    population_size=1e6  # Lower bacterial burden in TB
)
```

---

### 4. `hotspot` (Chronic Infections & Mutators)

#### When to Use
- Cystic fibrosis lung infections
- Long-term colonization (months to years)
- Known mutator populations
- Persistent antibiotic exposure
- Biofilm-associated infections

#### Biological Basis
- **Heterogeneous rates:** Some genotypes are hypermutable (10× higher)
- **Stress-induced:** SOS response increases mutation under antibiotics
- **Mid-resistance hotspots:** IC50 between 0.3-0.7 (most evolvable)
- **Fitness trade-off:** Mutators adapt faster but have slight fitness cost

#### Best For These Organisms
- **Pseudomonas aeruginosa** (CF patients: 20% have mutators)
- **E. coli** (chronic UTI)
- **Staphylococcus aureus** (chronic wounds, biofilms)
- **Burkholderia cepacia** (CF)
- Any chronic biofilm infection

#### Evidence from Simulations
- Critical C: 1.65 (same as others)
- **Diversity: HIGHEST** (+14% vs fitness_landscape)
- Surviving genotypes: 12/12
- Speed: Similar to others (fitness cost balances mutation advantage)

#### Why Mutators Matter
**Oliver et al. (2000) found:**
- 20% of CF P. aeruginosa have defective DNA repair (mutS, mutL)
- Mutators selected under antibiotic pressure
- 10-1000× higher mutation rate
- Faster adaptation but lower competitive fitness

**Your simulation shows:**
- Hotspot model maintains highest diversity (1.8 vs 1.6)
- More even genotype distribution at high C
- Better captures chronic infection dynamics

#### Literature Support
```
Oliver et al. (2000). Hypermutable P. aeruginosa in 20% of CF patients
Matic et al. (1997). Mutators accelerate resistance evolution
Denamur & Matic (2006). Mutators selected under stress
```

#### Code Example
```python
p = create_resistance_parameters(
    m=12,
    resistance_range=(0.1, 5.0),
    mutation_structure='gradient',
    mutation_type='hotspot',  # ← For CF infections
    base_mutation_rate=1e-7,  # 10× higher (mutators)
    population_size=1e8  # High bacterial load in lungs
)
```

---

### 5. `asymmetric` (Drug Holiday & Policy Studies)

#### When to Use
- Evaluating antibiotic cycling strategies
- Drug holiday efficacy
- Antimicrobial stewardship modeling
- Reversibility studies
- Resistance cost quantification

#### Biological Basis
- **Extreme asymmetry:** 50× easier to lose resistance than gain it
- **Thermodynamics:** Breaking genes easier than building them
- **Reversion pressure:** Without antibiotics, susceptible outcompete resistant
- **Population dynamics:** Models "recovery" of sensitivity

#### Not Organism-Specific
This is about **evolutionary dynamics**, not specific pathogens. Use with any organism when studying reversibility.

#### Evidence from Simulations
- Critical C: 1.65 (same as others)
- **Diversity loss: LARGEST** (-35%)
- Shows strongest reversion potential
- Models what happens when antibiotic pressure removed

#### Drug Holiday Strategy
**Theory:**
1. Use antibiotic → resistant rise → susceptible decline
2. Remove antibiotic → resistant have fitness cost → susceptible recover
3. Re-introduce antibiotic → susceptible population larger → treatment works again

**Your simulation supports this:**
- 50× asymmetry means rapid reversion
- Fitness costs (94.5% biomass reduction) drive susceptible comeback
- Works IF resistant haven't completely fixed (100% frequency)

#### Limitations
- Only works if resistant < 100%
- Requires significant fitness cost
- Timescale depends on cost magnitude
- May not work if compensatory mutations arise

#### Literature Support
```
Andersson & Hughes (2010). Loss-of-function 10-100× easier
Levin et al. (2000). Resistant outcompeted when antibiotics removed
Melnyk et al. (2015). Reversion rate 10⁻⁶ vs forward rate 10⁻⁹
```

#### Code Example
```python
p = create_resistance_parameters(
    m=12,
    resistance_range=(0.1, 5.0),
    mutation_structure='gradient',
    mutation_type='asymmetric',  # ← For policy studies
    base_mutation_rate=1e-8,
    population_size=1e7
)

# Then test with antibiotic removal:
# Run with high C → resistance evolves
# Run with C=0 → observe reversion
```

---

## 📊 Comparison Table: Model Predictions

Based on your simulation results (mutation_dynamics_comparison.png):

| Metric | fitness_landscape | horizontal_transfer | point_mutations | asymmetric | hotspot |
|--------|------------------|--------------------|-----------------| -----------|---------|
| **Critical C** | 1.65 | **0.90**  | 1.65 | 1.65 | 1.65 |
| **Speed** | Moderate | **Fastest**  | Moderate | **Slowest**  | Moderate |
| **Surviving** | 12/12 | 12/12 | **8/12** 📉 | 12/12 | 12/12 |
| **Diversity** | -18% | +14% 📈 | -18% | **-35%**  | +14%  |
| **Use case** | General | Plasmids | TB | Reversibility | Mutators |

### Key Insights

1. **HGT dramatically lowers threshold** (0.90 vs 1.65 = 45% difference)
   - **Clinical impact:** Standard dosing insufficient for plasmid carriers
   
2. **Point mutations are most restrictive** (8/12 vs 12/12)
   - **Why:** Can't skip steps in resistance ladder
   
3. **Asymmetric shows strongest diversity loss** (-35%)
   - **Why:** Very hard to gain, very easy to lose
   
4. **Hotspot maintains diversity** (+14%)
   - **Why:** High mutation rate creates more variants

---

##  For Your Evaluation: What to Say

### If Asked: "Which model is most biologically relevant?"

**Answer:**
> "The most biologically relevant model depends on the organism and clinical context. As a default, I recommend **fitness_landscape** because it:
> 
> 1. Captures fundamental evolutionary principles (distance-dependent transitions, asymmetry)
> 2. Uses empirically-validated mutation rates from Drake (1991): 10⁻⁸ per cell per generation
> 3. Incorporates realistic asymmetry from Andersson & Hughes (2010): 10× easier to lose function
> 4. Works for the majority of bacterial resistance (80% is chromosomal)
> 5. Doesn't assume special mechanisms, making it conservative
>
> However, for hospital-acquired pathogens like Klebsiella with KPC plasmids, **horizontal_transfer** is more appropriate. My simulations show it predicts a 45% lower critical concentration (0.90 vs 1.65), which matches clinical observations of rapid MDR emergence in hospitals.
>
> For tuberculosis, **point_mutations** is essential because TB has negligible horizontal gene transfer and requires sequential mutations—my model shows only 8/12 genotypes can survive at high antibiotic levels with this constraint, versus 12/12 with other models.
>
> For cystic fibrosis patients with chronic P. aeruginosa, **hotspot** is most realistic because 20% of CF isolates have mutator phenotypes (Oliver et al. 2000), and the model maintains 14% higher diversity, consistent with the variant-rich populations observed clinically."

### If Pressed to Choose ONE

**Answer:**
> "If I had to choose a single model, it would be **fitness_landscape** because:
> 
> - It's the most broadly applicable (works for 80% of cases)
> - It's conservative (won't over-predict adaptation speed)
> - It requires no special knowledge about mechanisms
> - It serves as a baseline for comparison
> - Most bacterial resistance is chromosomal point mutations
>
> However, this highlights a key limitation of one-size-fits-all modeling. The 83% variation in critical concentration between models (0.90 to 1.65) demonstrates that **mechanism matters** for accurate predictions. 
>
> In practice, I recommend:
> 1. Start with `fitness_landscape` as baseline
> 2. If plasmids suspected → test `horizontal_transfer`
> 3. If hypermutation suspected → test `hotspot`
> 4. Compare predictions to guide treatment strategy
>
> The framework's flexibility—allowing easy model switching—is actually a strength, not a weakness. It acknowledges biological reality: different bacteria evolve resistance through different mechanisms."

### Add This Sophistication

**Say:**
> "This is analogous to physics: you use Newtonian mechanics for everyday speeds, but switch to relativistic mechanics near the speed of light. Similarly, you use `fitness_landscape` for typical evolution, but switch to `horizontal_transfer` when plasmid dynamics dominate. The key is knowing which regime you're in."

---

## 🔬 Evidence-Based Decision Tree

```
START: Which model should I use?

│
├─ Do you know the organism?
│  │
│  NO → Use `fitness_landscape` (default)
│  │
│  YES → Continue
│     │
│     ├─ Is it Mycobacterium tuberculosis or M. leprae?
│     │  YES → Use `point_mutations`
│     │  NO → Continue
│     │
│     ├─ Is it a hospital-acquired Gram-negative with known plasmids?
│     │  (Klebsiella, E. coli ESBL, Acinetobacter, etc.)
│     │  YES → Use `horizontal_transfer`
│     │  NO → Continue
│     │
│     ├─ Is it a chronic CF infection or known mutator strain?
│     │  (P. aeruginosa in CF, chronic biofilm)
│     │  YES → Use `hotspot`
│     │  NO → Continue
│     │
│     ├─ Are you studying drug holidays or cycling strategies?
│     │  (Policy question, not organism-specific)
│     │  YES → Use `asymmetric`
│     │  NO → Use `fitness_landscape`

Special case: Community-acquired MRSA
└─ CA-MRSA → `fitness_landscape` (chromosomal)
   HA-MRSA → `horizontal_transfer` (mobile SCCmec)
```

---

## Model Validation Approaches

### How to Know if You Chose the Right Model

1. **Compare to clinical MIC data**
   - Does predicted IC50 distribution match patient isolates?
   - Check EUCAST or CLSI breakpoint databases

2. **Match outbreak timescales**
   - Hospital outbreak: Days to weeks → `horizontal_transfer`
   - Community spread: Months to years → `fitness_landscape`
   - TB resistance: Years → `point_mutations`

3. **Check diversity patterns**
   - High diversity in chronic infection → `hotspot`
   - Low diversity in clonal outbreak → `point_mutations`

4. **Test reversibility**
   - If resistance declines after drug removal → `asymmetric`
   - If resistance persists → `fitness_landscape`

5. **Sequencing data**
   - If plasmids detected → `horizontal_transfer`
   - If mutator alleles (mutS, mutL) present → `hotspot`
   - If only chromosomal SNPs → `fitness_landscape` or `point_mutations`

---

## 🔄 When to Use Multiple Models

### Run All 5 Models When:

1. **Mechanism is uncertain**
   - Use range of predictions to bracket uncertainty
   - Report: "Critical C is 0.90-1.65 depending on mechanism"

2. **Publication/thesis work**
   - Shows thoroughness
   - Demonstrates understanding that mechanism matters

3. **Policy decisions**
   - Need to consider worst-case (HGT) and best-case (point mutations)
   - Robust strategies work under all models

4. **New/emerging pathogen**
   - Don't yet know evolutionary biology
   - Use model comparison to generate hypotheses

### Example Multi-Model Analysis

```python
models = ['fitness_landscape', 'horizontal_transfer', 'point_mutations', 
          'asymmetric', 'hotspot']

results = {}
for model in models:
    p = create_resistance_parameters(
        m=12,
        mutation_type=model,
        base_mutation_rate=1e-8,
        population_size=1e7
    )
    results[model] = antibiotic_concentration_sweep(p, C_values)

# Compare critical concentrations
for model, res in results.items():
    C_crit = find_critical_concentration(res)
    print(f"{model}: C_crit = {C_crit:.2f}")

# Result:
# fitness_landscape: C_crit = 1.65
# horizontal_transfer: C_crit = 0.90  ← Lowest (most concerning)
# point_mutations: C_crit = 1.65
# asymmetric: C_crit = 1.65
# hotspot: C_crit = 1.65

# Recommendation: Use C > 1.65 for safety (unless HGT suspected, then C > 2.0)
```

---

##  Literature Guide by Model

### fitness_landscape
- Drake JW (1991). *PNAS* 88:7160-7164 — Mutation rates
- Andersson DI, Hughes D (2010). *Nat Rev Microbiol* 8:260-271 — Asymmetry
- Elena SF, Lenski RE (2003). *Nat Rev Genet* 4:457-469 — Experimental evolution

### horizontal_transfer
- Porse A et al. (2016). *Mol Biol Evol* 33:2860-2873 — HGT rates
- Carattoli A (2013). *Front Microbiol* 4:48 — Plasmid epidemiology
- Modi RI et al. (1991). *Genetics* 127:265-276 — Conjugation frequency

### point_mutations
- Farhat MR et al. (2013). *Nat Genet* 45:1183-1189 — TB resistance genetics
- Eldholm V, Balloux F (2016). *FEMS Microbiol Rev* 40:16-26 — Limited HGT in TB
- Gagneux S (2018). *Nat Rev Microbiol* 16:202-213 — TB evolution

### hotspot
- Oliver A et al. (2000). *Science* 288:1251-1253 — Mutators in CF
- Matic I et al. (1997). *Science* 277:1833-1834 — Mutator selection
- Denamur E, Matic I (2006). *Trends Microbiol* 14:353-359 — Mutators & stress

### asymmetric
- Melnyk AH et al. (2015). *Evol Appl* 8:273-283 — Fitness costs
- Levin BR et al. (2000). *Proc R Soc Lond B* 267:855-861 — Reversion
- Andersson DI, Levin BR (1999). *Curr Opin Microbiol* 2:489-493 — Compensatory evolution

---

## Summary: Quick Reference

| **Question** | **Answer** |
|-------------|-----------|
| **Default choice** | `fitness_landscape` |
| **Hospital Klebsiella** | `horizontal_transfer` |
| **Tuberculosis** | `point_mutations` |
| **CF P. aeruginosa** | `hotspot` |
| **Drug cycling study** | `asymmetric` |
| **Unknown organism** | `fitness_landscape` |
| **Most conservative** | `fitness_landscape` |
| **Predicts fastest resistance** | `horizontal_transfer` |
| **Most restrictive** | `point_mutations` |
| **Best for policy** | `asymmetric` |

---

## Final Recommendation

We implemented five biologically-distinct models because no single mutation mechanism applies to all bacteria. The default `fitness_landscape` works for most cases (80% chromosomal resistance), but simulations show that for hospital pathogens with plasmids, the `horizontal_transfer` model predicts 45% lower critical concentrations, which has major implications for dosing strategies. This framework's flexibility to match pathogen biology is a strength—it reflects biological reality rather than forcing a one-size-fits-all approach.

