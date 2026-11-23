# How the synthetic portfolio is generated

`src/ews/generate.py` builds a seeded monthly panel of 3,000 corporate counterparties from January 2021 to December 2025. No real bank, counterparty or customer data is used.

## Counterparties

| Attribute | Values |
|---|---|
| Segment | SME 55%, mid-corporate 25%, large corporate 8%, property developer 12% |
| Sector | 10 sectors (property developers are all in real estate development) |
| Region | 8 Moroccan regions, weighted towards Casablanca-Settat |
| Exposure | Log-normal around a segment median: SME MAD 3m, mid-corporate 25m, large corporate 150m, developer 60m |

## Latent credit quality

Each counterparty has a long-run quality level, shifted by segment (large corporates healthier, developers weaker). Monthly quality follows an AR(1) process around that level (persistence 0.93), with:
- **sector cycles:** a 2021 hit to tourism, the 2022 drought in agriculture, and a construction and real-estate slowdown from mid-2023 to early 2025;
- **distress episodes:** random spells of 6-15 months in which quality drifts down, more frequent for weaker names.

## What the bank observes

| Data | Frequency | Delay | How it depends on quality |
|---|---|---|---|
| Overdraft use, limit-breach days, payment incidents, account turnover | monthly | 1 month | logistic and Poisson links, with noise |
| Days past due | monthly | 1 month | sporadic arrears when quality is low; arrears build up in the 3 months before default |
| Financial statements (leverage, interest coverage, current ratio, revenue growth) | annual | published 6 months after year end | log-normal links to the fiscal-year average |
| Internal rating (1 best to 9 worst, D = default) | annual review | uses the last **audited** fiscal year, so 6-18 months old | plus a persistent rating-model error per name; ad hoc downgrade to 8 after 30+ days past due |

## Defaults

A monthly hazard rises with low quality: hazard = 1 / (1 + exp(6.4 + 1.5 q)). Defaulted names stay in the book as non-performing and are written off 24 months later. The resulting annual default rate is about 3%, with property developers and SMEs above large corporates, and a rise in construction and real-estate defaults in 2024-2025.

## Design choices that matter for the results

- **Ratings look backwards.** They are set from audited statements, which is how annual credit reviews work. This is why account behaviour can lead the rating.
- **Defaults are random given quality.** Some defaults come with little warning, so no model can catch everything.
- **Realism checks** (in the tests): annual default rate 2-5%; NPL ratios of a few percent to low double digits; 55-70% of names keep their grade over a year.

## Files written to `data/` (seed 42)

| File | Content |
|---|---|
| `counterparties.csv` | segment, sector, region, onboarding month, annual review month |
| `monthly_panel.csv.gz` | status, exposure, behaviour, days past due, internal rating, per counterparty per month |
| `financials.csv` | annual statements with the month they become available |
| `ground_truth.csv` | default month and long-run quality (evaluation only, never a model input) |
| `latent.csv.gz` | monthly latent quality and distress flag (evaluation only) |
