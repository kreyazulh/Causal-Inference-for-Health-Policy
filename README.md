# Causal-Inference-for-Health-Policy

A framework for evaluating causal inference methods on health policy interventions using the POCME (Performance-Oriented Causal Method Evaluation) framework.

## Overview

This repository implements Meta-DML (Meta-Learning Enhanced Double Machine Learning) and compares it against 10 other causal inference methods across 3,247 indicator-intervention pairs from Bangladesh, Philippines, and Zimbabwe.

## Setup

Create virtual environment and activate:
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Install dependencies:
```powershell
pip install -r requirements.txt
```

## Usage

Run main analysis:
```powershell
python main.py
```

Generate visualizations:
```powershell
python extract_cikm_findings.py
```

## Domain Constraints

The following epidemiologically-plausible bounds are enforced for relative effect estimates:

| Health Indicator | Min Effect (%) | Max Effect (%) |
|-----------------|----------------|----------------|
| Mortality rate, infant (per 1,000 live births) | -80 | 10 |
| Life expectancy at birth, total (years) | -10 | 15 |
| Maternal mortality ratio (per 100,000 live births) | -80 | 10 |
| Immunization, measles (% of children ages 12-23 months) | -20 | 80 |
| Prevalence of undernourishment (% of population) | -70 | 15 |
| Mortality rate, under-5 (per 1,000 live births) | -80 | 10 |
| Incidence of tuberculosis (per 100,000 people) | -80 | 20 |
| Hospital beds (per 1,000 people) | -50 | 200 |

## Policy Timelines

### Bangladesh
- **1972**: National Health Policy in first Five Year Plan
- **1976**: Population Control and Family Planning Program
- **1978**: Adoption of Alma-Ata Declaration principles
- **1982**: National Drug Policy implementation
- **1988**: National Health Policy established
- **1993**: National Immunization Program expansion
- **1998**: Health and Population Sector Programme (HPSP)
- **2000**: Bangladesh Integrated Nutrition Project
- **2003**: Health, Nutrition and Population Sector Program
- **2005**: National HIV/AIDS Policy
- **2008**: Revitalized National Health Policy
- **2011**: Health Population and Nutrition Sector Development Program
- **2016**: Health Care Financing Strategy
- **2021**: Bangladesh Health Sector Strategy 2022-2031

### Philippines
- **1972**: Implementation of Philippine Medical Care Act (Medicare)
- **1976**: Compulsory Basic Immunization Program (PD 996)
- **1978**: Revised Medicare Act with New Society policies (PD 1519)
- **1980**: Primary Health Care approach adoption post-Alma Ata
- **1988**: Generics Act implementation
- **1991**: Local Government Code - Health service devolution
- **1993**: Doctors to the Barrios Initiative launch
- **1995**: National Health Insurance Act - PhilHealth creation
- **1999**: Health Sector Reform Agenda (HSRA) 1999-2004
- **2005**: FOURmula One for Health strategy 2005-2010
- **2008**: Cheaper Medicines Act and health financing reforms
- **2010**: Aquino Health Agenda - Universal Health Care focus
- **2012**: Sin Tax Reform Law for health financing
- **2016**: Philippine Health Agenda 2016-2022
- **2017**: FOURmula One Plus for Health (F1+) 2017-2022
- **2019**: Universal Health Care Act - comprehensive reform
- **2021**: Post-pandemic health system strengthening

### Zimbabwe
- **1980**: Independence and Primary Health Care adoption
- **1982**: Rural Health Centers expansion program
- **1988**: Essential Drug List implementation
- **1990**: Economic Structural Adjustment Program health impacts
- **1996**: Health Services Fund introduction with user fees
- **1997**: National Health Strategy 1997-2007
- **2000**: Land reform and donor relations crisis
- **2003**: National AIDS Trust Fund establishment
- **2008**: Health system collapse and dollarization
- **2009**: National Health Strategy 2009-2013 - recovery focus
- **2013**: Health Development Fund with international partners
- **2016**: National Health Strategy 2016-2020 "Leaving No One Behind"
- **2018**: Post-Mugabe health sector recovery initiatives
- **2021**: National Health Strategy 2021-2025
- **2023**: Health Resilience Fund launch for UHC

## Data Sources

- Health indicators: World Bank Health Data
- Policy interventions: Manually curated from government documents and WHO reports
- Validation data: Available at [World Bank Data](https://data.worldbank.org)

## Methods Evaluated

1. **Meta-DML** (our method)
2. DoubleML
3. BART (Bayesian Additive Regression Trees)
4. Synthetic Control Method (SCM)
5. Propensity Score Matching (PSM)
6. Augmented Synthetic Control Method (ASCM)
7. CausalImpact
8. Interrupted Time Series (ITS)
9. CausalForests
10. Difference-in-Differences (DiD)
