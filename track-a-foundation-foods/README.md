# Diet Optimization — USDA Foundation Foods

Multi-objective analysis and machine learning on 369 USDA Foundation Foods, combining nutrient data, environmental footprint (CO2), and retail prices to identify optimal whole foods and diets.

---

## Project Overview

This project builds a full data science pipeline from raw USDA data to interpretable multi-objective optimization and supervised ML. The central question is: **what whole foods minimize cost and environmental impact while maximizing nutritional quality?**

Three data sources are merged using two-stage semantic matching (bi-encoder + cross-encoder reranking), since no shared key exists between the USDA, French LCA, and USDA price databases. The pipeline then applies Pareto optimality, TOPSIS ranking, and Random Forest + XGBoost classification and regression with full SHAP attribution.

---

## Notebooks

| Notebook | Description |
|---|---|
| `A1_foundation_preparation.ipynb` | Data pipeline: load USDA Foundation Foods, semantic matching to AGRIBALYSE (CO2) and FMAP (price), three-layer CO2 correction, data quality fixes |
| `A2_foundation_eda_clustering.ipynb` | EDA, PCA, K-Means + Hierarchical clustering, cluster DNA heatmap, cluster interpretation |
| `A3_foundation_pareto.ipynb` | 3-objective Pareto optimality (price, CO2, protein) on a per-1000-kcal functional unit, sensitivity analysis across 4 weighting scenarios |
| `A6_topsis_full.ipynb` | Full TOPSIS ranking across 20 nutritional objectives, 1M random diet sampling (product-based and group-diversity-constrained), sensitivity analysis |
| `A7_ml_food_classification.ipynb` | **Track D:** food group classification from nutrient profile alone (RF + XGBoost, OOF evaluation, SHAP). **Track A:** TOPSIS score regression with SHAP attribution, underrated/overrated food analysis |

> Notebooks A4 and A5 are intermediate development steps (multi-nutrient Pareto and hybrid TOPSIS/Pareto) retained for reference but superseded by A6.

---

## Data Sources

| Dataset | Description | Source |
|---|---|---|
| **USDA Foundation Foods** | 369 minimally processed whole foods with detailed nutrient profiles (22 nutrients), released by USDA ARS | [FoodData Central](https://fdc.nal.usda.gov/download-datasets) — `FoundationFoods.json` |
| **AGRIBALYSE 3.2** | French LCA database with CO2-equivalent footprints (kg CO2-eq/kg product) for ~2500 food products, maintained by ADEME | [Recherche Data Gouv](https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/XTENSJ) — `AGRIBALYSE3.2_Tableur produits alimentaires.xlsx` |
| **FMAP** | USDA ERS retail price database, ~90 broad food categories, national averages 2016–present ($/kg) | [USDA ERS](https://www.ers.usda.gov/data-products/food-at-home-monthly-area-prices) — `FMAP.xlsx` |

---

## Methods

### Semantic Matching (A1)
Direct string matching between USDA and AGRIBALYSE/FMAP names fails due to language differences (English vs. French) and naming conventions. A two-stage pipeline is used:
- **Stage 1 — Bi-encoder** (`all-MiniLM-L6-v2`): dense embedding retrieval of top-10 candidates per food
- **Stage 2 — Cross-encoder** (`stsb-distilroberta-base`): reranks the top-10 pairs for accurate relevance scoring

A three-layer CO2 correction handles systematic match failures: manual overrides for known bad matches → suspicious value detection (>60 kg CO2-eq/kg) → food-group mean fallback for low-confidence matches.

### Functional Unit (A3)
All objectives are expressed per 1000 kcal rather than per kg of food. This prevents systematic bias against water-rich foods (vegetables, fish) and ensures comparisons answer a consistent question: *what does it cost in money, CO2, and protein to get 1000 kcal from this food?*

### Pareto Optimality (A3)
A food is Pareto-optimal if no other food is simultaneously cheaper, lower in CO2, and higher in protein. Objectives are min-max normalized before the dominance check. Sensitivity analysis across four named weighting scenarios (Balanced, Eco-conscious, High Protein Budget, Budget) identifies robustly optimal foods that appear on the front regardless of priorities.

### TOPSIS (A6)
With 20 objectives, Pareto optimality breaks down — ~80% of foods become non-dominated (curse of dimensionality). TOPSIS collapses all objectives to a single 0–1 score: distance to the ideal solution divided by total spread. Higher score = closer to perfect on all 20 objectives simultaneously. Applied to both individual foods and 1,000,000 randomly sampled 2000-kcal diets.

### Machine Learning (A7)
**Classification:** Random Forest + XGBoost predict food group (7 classes) from 22 nutrient density features. All evaluation uses out-of-fold (OOF) cross-validation predictions via `cross_val_predict` — each food is predicted by a model that never saw it during training. Full-data fit used exclusively for SHAP attribution.

**Regression:** Random Forest predicts TOPSIS score from nutrients. SHAP decomposes the score into per-nutrient contributions in a nonlinear, interaction-aware way. Foods with large positive residuals (actual >> predicted) are nutritionally unusual — their quality comes from synergistic combinations the model underestimates.

---

## Key Results

- **Legumes dominate the 3-objective Pareto front** (A3): low price, low CO2, high protein per 1000 kcal — robust across all four weighting scenarios
- **TOPSIS top foods** (A6): fatty fish, eggs, and dark leafy vegetables consistently rank highest across all 20 objectives under equal weighting
- **Diet sampling** (A6): group-diversity-constrained diets (one product per food group) achieve systematically higher TOPSIS scores than unconstrained random diets, confirming that dietary variety is nutritionally beneficial beyond individual food selection
- **Classification** (A7): macro F1 well above baseline — most food groups are highly separable from nutrient density alone; the hardest classes to separate reveal genuine nutritional overlap (e.g., fats group confused with nuts/seeds)
- **SHAP** (A7): protein density and fiber are the strongest positive TOPSIS drivers; sodium and saturated fat the strongest negative drivers — consistent across both the classifier and regressor SHAP rankings

---

## Repository Structure

```
diet-optimization-usda/
├── A1_foundation_preparation.ipynb
├── A2_foundation_eda_clustering.ipynb
├── A3_foundation_pareto.ipynb
├── A6_topsis_full.ipynb
├── A7_ml_food_classification.ipynb
├── data/
│   ├── FoundationFoods.json          # USDA (download separately)
│   ├── AGRIBALYSE3.2_...xlsx         # AGRIBALYSE (download separately)
│   └── FMAP.xlsx                     # USDA ERS (download separately)
└── README.md
```

Intermediate CSVs (`foundation_analysis_ready.csv`, `foundation_clustered.csv`, etc.) are generated by running notebooks in order A1 → A2 → A3 → A6 → A7.

---

## Setup

```bash
pip install numpy pandas matplotlib seaborn scikit-learn sentence-transformers xgboost shap openpyxl
```

Python 3.9+ recommended. `sentence-transformers` downloads model weights (~90 MB) on first run. All other dependencies are standard scientific Python.

Run notebooks in order: **A1 → A2 → A3 → A6 → A7**. Each notebook reads the CSV output of the previous one.

---

## Notes on Data Approximations

- **CO2 matching** to AGRIBALYSE is semantic — not all matches are product-level. Low-confidence matches fall back to food-group means. Match confidence (`bi_encoder_conf`) is retained in all output CSVs.
- **Price matching** to FMAP covers only ~90 broad categories. Price is a group-level approximation, not a product-level measurement. Cheese prices are manually corrected (FMAP maps most cheeses to the milk category).
- Both approximations are intentional design choices given the data landscape — they are documented in A1 and propagated transparently through the pipeline.
