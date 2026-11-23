# Early-warning system: results

Synthetic corporate portfolio (3,000 counterparties, monthly 2021-2025). Models trained on snapshots
from July 2021 to December 2022 and tested out of time on 2024. Early-warning figures cover 2025 defaults
scored monthly from January 2024. Means over 3 generator seeds (42, 1, 2).

## Model comparison at equal alert volume

Each model flags the top 4% of the performing book every month, about the volume the
current process (rating 8-9 or 30+ days past due) already puts on watch.

| Model | AUC | PR-AUC | Defaults caught at capacity | Defaults flagged 6+ months ahead | Alerts per month |
|---|---|---|---|---|---|
| Current process | 0.800 | 0.206 | 35% | 29% ± 5 | 103 |
| Expert rules | 0.816 | 0.192 | 30% | 33% ± 3 | 106 |
| Logistic scorecard | 0.822 | 0.216 | 35% | 32% ± 2 | 106 |
| Gradient boosting | 0.812 | 0.272 | 36% | 33% ± 5 | 106 |

- On average the behaviour-based models flag **more defaults early** (6+ months before default) than the
  current process at the same alert volume (32-33% against 29%), though the seed-to-seed spread is wide. Gradient boosting ranks the riskiest names best (highest PR-AUC).
- Overall discrimination gains are **modest** (AUC +0.01 to +0.02). Ratings and arrears already carry
  much of the signal; the value of account behaviour is in timing.
- Expert rules are transparent but catch fewer defaults at this capacity.
- Score stability between training and test populations: PSI = 0.005 (below 0.10 = stable).

## Calibration (logistic scorecard, 2024 test year)

| Decile | Mean predicted PD | Observed default rate | Names x months |
|---|---|---|---|
| 1 | 0.16% | 0.03% | 3239 |
| 2 | 0.37% | 0.37% | 3238 |
| 3 | 0.61% | 1.45% | 3238 |
| 4 | 0.89% | 1.42% | 3238 |
| 5 | 1.23% | 0.93% | 3239 |
| 6 | 1.72% | 2.01% | 3238 |
| 7 | 2.36% | 2.32% | 3238 |
| 8 | 3.38% | 2.63% | 3238 |
| 9 | 5.33% | 4.23% | 3238 |
| 10 | 18.29% | 14.97% | 3239 |

## Alert list, 2025-12

104 names flagged; 23 of them are **not yet on the current watch list**: these are the
early warnings an analyst would review first. The 10 riskiest of them:

| Counterparty | Segment | Sector | Rating | 12-month PD | Already on watch | Main reasons |
|---|---|---|---|---|---|---|
| CP02961 | SME | Wholesale & retail trade | 7 | 23.4% | no | internal rating (9 = worst) high; days past due high; overdraft use high |
| CP01453 | SME | Energy & utilities | 7 | 22.8% | no | internal rating (9 = worst) high; overdraft use high; days past due high |
| CP02001 | Property developer | Real estate development | 7 | 22.4% | no | internal rating (9 = worst) high; overdraft use high; net debt / EBITDA high |
| CP02059 | Mid-corporate | Wholesale & retail trade | 7 | 20.9% | no | internal rating (9 = worst) high; overdraft use high; days past due high |
| CP00430 | SME | Wholesale & retail trade | 7 | 19.5% | no | internal rating (9 = worst) high; overdraft use high; worst arrears in 6 months high |
| CP00177 | SME | Tourism & hospitality | 7 | 19.5% | no | internal rating (9 = worst) high; overdraft use high; net debt / EBITDA high |
| CP00392 | SME | Real estate services | 6 | 19.5% | no | internal rating (9 = worst) high; days past due high; overdraft use high |
| CP00136 | SME | Business services | 7 | 18.3% | no | internal rating (9 = worst) high; overdraft use high; interest coverage low |
| CP00741 | Property developer | Real estate development | 7 | 18.1% | no | internal rating (9 = worst) high; overdraft use high; net debt / EBITDA high |
| CP02114 | Mid-corporate | Business services | 7 | 17.9% | no | internal rating (9 = worst) high; overdraft use high; months with overdraft near limit high |

Full list: `outputs/alert_list_2025-12.csv`.

## Portfolio quality

- NPL ratio, December 2025: Large corporate 3.3%, Mid-corporate 3.9%, Property developer 12.5%, SME 6.9%.
- Cost of risk (expected-loss proxy), % of average exposure: 2022 1.62%, 2023 0.97%, 2024 1.23%, 2025 1.42%.
- Rating stability: 63% of names rated 3-6 keep the same grade over a year.
- Figures 4-7 in `outputs/figures/` show NPL by segment, cost of risk by sector, the migration matrix and
  exposure by rating bucket.

## Caveats

- The portfolio is synthetic. Its behaviour, rating and default mechanics are modelled assumptions (see
  `docs/DATA_GENERATION.md`), so the numbers show the method, not the performance on a real bank's book.
- About 85 defaults per test year is a small sample; differences of a few points between models are within
  seed-to-seed variation (shown as ± one standard deviation).
