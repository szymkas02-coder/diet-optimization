# Diet Optimization — USDA Food Data Science

A two-part data science project combining multi-objective optimization, TOPSIS ranking, and interpretable machine learning on USDA food databases.

## Projects

### [Track A — Foundation Foods](./track-a-foundation-foods/)
Multi-objective optimization and ML on 369 USDA Foundation Foods (whole raw ingredients). Merges USDA nutrient data, AGRIBALYSE CO2 footprints, and FMAP retail prices using two-stage semantic matching. Identifies Pareto-optimal and TOPSIS-optimal foods and diets across 20 nutritional objectives. Includes food group classification and TOPSIS score regression with full SHAP attribution.

**Notebooks:** A1 (data pipeline) → A2 (EDA + clustering) → A3 (Pareto) → A6 (TOPSIS + diet sampling) → A7 (ML + SHAP)

### [Track B — FNDDS Mixed Dishes](./track-b-fndds/)
Clustering, recommendation engine, and ML on 5431 USDA FNDDS mixed dishes (real composite meals). Finds nutritionally superior substitutes for queried meals using semantic search + nutrient distance filtering. Classifies meal categories from nutrient profiles alone using Random Forest + XGBoost with full OOF evaluation and SHAP attribution.

**Notebooks:** B1 (data pipeline + recommendation engine) → B2 (ML + SHAP)

## Shared Methodology
Both projects use the same semantic matching pipeline (bi-encoder + cross-encoder reranking) to join USDA nutrient data with AGRIBALYSE CO2 footprints and FMAP prices. All ML evaluation uses out-of-fold cross-validation predictions. SHAP TreeExplainer is used for attribution throughout.

## Data Sources
| Dataset | Used in | Link |
|---|---|---|
| USDA Foundation Foods JSON | Track A | [FoodData Central](https://fdc.nal.usda.gov/download-datasets) |
| USDA FNDDS 2021–2023 JSON | Track B | [FoodData Central](https://fdc.nal.usda.gov/download-datasets) |
| AGRIBALYSE 3.2 | Both | [Recherche Data Gouv](https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/XTENSJ) |
| FMAP prices | Both | [USDA ERS](https://www.ers.usda.gov/data-products/food-at-home-monthly-area-prices) |

## Setup
```bash
pip install numpy pandas matplotlib seaborn scikit-learn sentence-transformers xgboost shap openpyxl scipy
```
Run Track A notebooks in order A1→A2→A3→A6→A7. Track B notebooks in order B1→B2. Raw data files must be downloaded separately (links above) and placed in the respective `data/` folder.
