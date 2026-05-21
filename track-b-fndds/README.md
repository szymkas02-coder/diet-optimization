# FNDDS Meal Analysis — Healthier Substitutes & Nutrient Classification

Clustering, recommendation engine, and machine learning on 5431 USDA FNDDS mixed dishes — real composite meals Americans eat — to find nutritionally superior substitutes and classify meal categories from nutrient profiles alone.

---

## Project Overview

While the Track A project works with raw whole ingredients (Foundation Foods), this project operates on **FNDDS (Food and Nutrient Database for Dietary Studies)** — the database of real mixed dishes recorded in US dietary recall surveys: pizzas, burgers, stir-fries, casseroles, and thousands of others.

The core questions are:
1. **Given a meal someone is eating, what is a nutritionally better alternative that is similar enough to be a realistic substitution?** (B1)
2. **Can a model identify a meal's food category using only its nutrient density profile — no names, no metadata?** (B2)

---

## Notebooks

| Notebook | Description |
|---|---|
| `B1_fndds_recommendation_engine.ipynb` | Data pipeline (JSON parsing, nutrient extraction, semantic CO2/price matching), K-Means + Hierarchical clustering, healthier substitute recommendation engine with dietary and environmental win flags |
| `B2_fndds_ml_classification.ipynb` | Food category classification from nutrient profile alone (RF + XGBoost, OOF evaluation, learning curves, SHAP global and per-class attribution, pairwise SHAP comparisons, misclassification deep dive) |

---

## Data Source

| Dataset | Description | Source |
|---|---|---|
| **USDA FNDDS 2021–2023** | 5431 mixed dishes with full nutrient profiles from US dietary recall surveys. Each dish has a WWEIA (What We Eat In America) category code. | [FoodData Central](https://fdc.nal.usda.gov/download-datasets) — `FNDDS.json` |
| **AGRIBALYSE 3.2** | CO2-equivalent footprints for ~2500 food products (best-effort matching to mixed dishes) | [Recherche Data Gouv](https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/XTENSJ) |
| **FMAP** | USDA ERS retail prices, ~90 broad categories | [USDA ERS](https://www.ers.usda.gov/data-products/food-at-home-monthly-area-prices) |

---

## Methods

### B1 — Recommendation Engine

#### Nutrient Extraction & Preprocessing
FNDDS JSON is parsed to extract 22 target nutrients per dish. Energy is computed via Atwater factors (4/4/9 kcal/g for protein/carbs/fat) where missing. All nutrients are normalized to a per-100-kcal basis for energy-controlled comparison.

#### Clustering
StandardScaler → PCA (5 components) → K-Means (K=12) + Agglomerative Hierarchical (K=12, Ward linkage). K=12 is chosen over A2's K=7 because FNDDS has ~15× more foods with substantially higher category diversity (soups, ethnic dishes, fast food, desserts, beverages all co-exist).

#### Substitute Recommendation — Three-Stage Pipeline
Given a query meal name (e.g., "Double cheeseburger"):

1. **Semantic candidate retrieval** — bi-encoder (`all-MiniLM-L6-v2`) encodes the query and retrieves the top-200 semantically similar dishes from all 5431 FNDDS names. Pool is intentionally large to avoid filtering out nutritionally better options with different terminology.
2. **Nutrient distance filtering** — each candidate's nutrient density profile is compared against the query using Euclidean distance in PCA-reduced space. Candidates outside the similarity threshold are dropped, ensuring substitutes are genuinely similar meals.
3. **Dietary win ranking** — candidates ranked by per-portion improvements across five key metrics: potassium and fiber (maximize), saturated fat and sodium (minimize), and calorie density. All comparisons use a fixed 500-kcal reference portion so results are energy-equivalent.

When CO2 and price data are available, candidates with >20% CO2 reduction are flagged `[ECO WIN]` and >10% price reduction as `[BUDGET WIN]`.

#### CO2 and Price — Important Caveat
Mapping CO2 footprints and retail prices onto 5431 mixed dishes is **substantially more approximate** than for Foundation Foods. FMAP has ~90 broad categories and AGRIBALYSE ~2500 products, while FNDDS has 5431 dishes with complex ingredient combinations. Most FNDDS dishes receive group-level CO2 means rather than product-specific LCA values. CO2 and price figures should be interpreted as order-of-magnitude estimates useful for relative comparisons within a category, not as precise life-cycle assessments.

### B2 — Meal Category Classification

#### Food Group Mapping
FNDDS uses WWEIA (What We Eat In America) categories — a fine-grained ~172-category system. These are manually aggregated into 18 interpretable broader groups based on the numeric WWEIA prefix structure:

| Prefix | Group |
|---|---|
| 1xxx | dairy, plant_based_milk |
| 2xxx | beef_pork_lamb, poultry, fish_shellfish, eggs, processed_meat, legumes_nuts |
| 3xxx | mixed_dishes, fast_food, soups |
| 4xxx | grains_breads |
| 5xxx | snacks, sweets_desserts |
| 6xxx | fruits, vegetables |
| 7xxx | beverages, alcohol |
| 8xxx | fats_condiments |

#### Machine Learning Pipeline
StandardScaler → Random Forest + XGBoost, stratified 5-fold CV, macro F1 scoring. Groups with fewer than 30 samples are dropped. `class_weight="balanced"` compensates for the dominant `mixed_dishes` class (~700+ foods).

#### Evaluation Philosophy
All metrics — confusion matrix, per-class accuracy, classification report, confidence analysis — are computed from **out-of-fold (OOF) cross-validation predictions** via `cross_val_predict`. Each meal is predicted by a model that never saw it during training. The full-data fit is used exclusively for SHAP attribution (TreeExplainer requires a single fitted model).

---

## Key Results

### B1 — Recommendation Engine
- The engine successfully identifies nutritionally superior substitutes for queried meals while preserving culinary similarity
- Per-500-kcal portion comparison avoids the water-content distortion that would occur with per-100g or per-kg comparisons
- CO2 and price deltas are available for ~35–40% of dish pairs (high-confidence semantic matches); the remainder show nutrient-only comparisons

### B2 — Classification
- Random Forest and XGBoost both substantially exceed the majority-class baseline macro F1 (~0.01)
- **Most separable groups:** dairy, eggs, beverages, fruits, vegetables — nutritionally very distinct from other categories
- **Hardest groups:** `mixed_dishes` and `fast_food` — the highest confusion pair in the OOF matrix, confirming these categories are nutritionally nearly indistinguishable from nutrient density alone. This is a genuine finding: fast food and home-cooked mixed dishes have converged nutritionally in the FNDDS data
- **SHAP:** fiber, protein, sodium, carbohydrate, and vitamin C are the most globally discriminative nutrients; per-class SHAP reveals that sodium and saturated fat strongly define `fast_food` and `processed_meat`, while fiber and potassium define `vegetables` and `fruits`
- **Misclassification deep dives:** the most surprising OOF misclassifications reveal meals that genuinely bridge category boundaries — a nutritionally informative finding beyond the accuracy metric

---

## Repository Structure

```
fndds-meal-analysis/
├── B1_fndds_recommendation_engine.ipynb
├── B2_fndds_ml_classification.ipynb
├── data/
│   ├── FNDDS.json                    # USDA FNDDS 2021-2023 (download separately)
│   ├── AGRIBALYSE3.2_...xlsx         # AGRIBALYSE (download separately, optional)
│   └── FMAP.xlsx                     # USDA ERS (download separately, optional)
└── README.md
```

B2 requires only `FNDDS.json`. B1 additionally uses AGRIBALYSE and FMAP for CO2/price enrichment — the recommendation engine works without them (nutrient-only mode).

---

## Setup

```bash
pip install numpy pandas matplotlib seaborn scikit-learn sentence-transformers xgboost shap openpyxl scipy
```

Python 3.9+ recommended. `sentence-transformers` downloads model weights (~90 MB) on first run (B1 only — B2 does not use sentence transformers).

Run **B1 first** if you want CO2/price data available in the recommendation engine output. B2 is fully self-contained and can be run independently from `FNDDS.json` alone.

---

## Relationship to Track A

This project is a companion to the [diet-optimization-usda](../diet-optimization-usda) project (Track A). The key differences:

| | Track A (Foundation Foods) | Track B (FNDDS) |
|---|---|---|
| **Foods** | 369 raw whole ingredients | 5431 real composite meals |
| **Goal** | Identify optimal whole foods and diets | Find better meal substitutes; classify meal types |
| **CO2/price quality** | Product-level matches (~70% high-confidence) | Group-level approximation (~35–40% high-confidence) |
| **ML task** | Classification + TOPSIS regression | Classification only |
| **Hardest ML boundary** | Fats vs. nuts/seeds | Fast food vs. mixed dishes |

The semantic matching pipeline, nutrient normalization approach, and SHAP evaluation methodology are shared between both projects, defined once in Track A (A1) and adapted in Track B (B1, B2).

---

## License

[MIT](../LICENSE) © 2026 szymkas02
