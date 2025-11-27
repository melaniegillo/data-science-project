# AI Usage Log

This document tracks all AI-generated or AI-assisted code in this project, as required by the course policy.

## Project Overview
- **Project**: Bitcoin Value-at-Risk Forecasting
- **Research Question**: Does incorporating information from the VIX improve the accuracy of one-day Value-at-Risk forecasts for Bitcoin, compared with standard Historical and Monte Carlo VaR models, across different rolling estimation windows?
- **AI Tool Used**: Claude Code (Claude Sonnet 4.5)

---

## Phase 1: Infrastructure Setup (Date: 2025-11-27)

### Component: pyproject.toml
- **AI Tool:** Claude Code
- **Prompt:** "Create pyproject.toml for Bitcoin VaR forecasting project with dependencies: pandas, numpy, scipy, matplotlib, openpyxl, pytest, ruff"
- **Generated Code:** Complete pyproject.toml file (~40 lines)
- **Modifications:** None - used as generated
- **Understanding:** Yes, I understand the pyproject.toml format, dependency specification, and build system configuration

### Component: .gitignore updates
- **AI Tool:** Claude Code
- **Prompt:** "Update .gitignore to exclude results/, src/CSV_*, and src/BTC_Kupiec_*.csv"
- **Generated Code:** Additional gitignore patterns (~3 lines)
- **Modifications:** None - added to existing gitignore
- **Understanding:** Yes, I understand gitignore patterns and why we exclude generated results

### Component: tests/ directory and AI_LOG.md template
- **AI Tool:** Claude Code
- **Prompt:** "Create tests directory with __init__.py and AI_LOG.md template"
- **Generated Code:** Directory structure and AI_LOG.md template (~50 lines)
- **Modifications:** Customized AI_LOG.md template for this specific project
- **Understanding:** Yes, I understand the purpose of test organization and documenting AI usage

---

## Phase 2: Configuration and Data Loader Improvements (Date: 2025-11-27)

### Component: src/config.py
- **AI Tool:** Claude Code
- **Prompt:** "Create simple configuration module with paths, rolling windows, confidence levels, and model parameters using pathlib"
- **Generated Code:** Complete config.py (~80 lines)
- **Modifications:** None - used as generated
- **Understanding:** Yes, I understand pathlib usage, configuration patterns, and why we centralize constants

### Component: src/data_loader.py (Refactored)
- **AI Tool:** Claude Code
- **Prompt:** "Refactor data_loader.py to use config, add error handling, column validation, and create lagged VIX for forecasting"
- **Generated Code:** Complete refactored data_loader.py (~180 lines)
- **Modifications:** None - used as generated
- **Understanding:** Yes, I understand:
  - Error handling with try/except and custom exceptions
  - Why we need lagged VIX (line 154): Uses VIX(t-1) to forecast at time t, avoiding look-ahead bias
  - Data validation to ensure required columns exist
  - Pandas operations for merging, feature engineering, and data cleaning

### Key Improvement: Lagged VIX Feature
- **Critical Addition:** `data['VIX_lag1'] = data['VIX_decimal'].shift(1)` (line 154)
- **Purpose:** Enables true forecasting by using VIX from previous day
- **Why Important:** Original code didn't have this; the VIX model was using contemporaneous VIX (look-ahead bias)
- **Understanding:** Fully understand the forecasting logic and why lagging is essential

---

## Phase 3: Model Refactoring and Critical VIX Bug Fix (Date: 2025-11-27)

### Component: src/models_vix_regression.py (CRITICAL FIX)
- **AI Tool:** Claude Code
- **Prompt:** "Refactor VIX regression model to function-based approach AND fix the critical lagging bug on line 48"
- **Generated Code:** Complete refactored models_vix_regression.py (~165 lines)
- **Critical Fix:** Line 111 changed from `x[i]` to `x[i-1]` for true forecasting
- **Modifications:** None - used as generated
- **Understanding:** Yes, I FULLY understand:
  - **THE BUG:** Original code used `sigma_ann = intercept + slope * x[i]` (contemporaneous VIX)
  - **THE FIX:** New code uses `sigma_ann = intercept + slope * x[i-1]` (lagged VIX)
  - **WHY IT MATTERS:** Using x[i] is look-ahead bias - using future information to forecast the present
  - **CORRECT APPROACH:** At time i, we only have data up to time i-1, so must use VIX(i-1) to forecast
  - This makes the VIX model a TRUE forecasting model, comparable to Historical and Monte Carlo
  - Added extensive comments (lines 104-111) explaining the fix

### Component: src/models_historical.py (Refactored)
- **AI Tool:** Claude Code
- **Prompt:** "Refactor historical VaR model to function-based approach with docstrings and config usage"
- **Generated Code:** Complete refactored models_historical.py (~140 lines)
- **Modifications:** None - used as generated
- **Understanding:** Yes, I understand:
  - Empirical quantile approach (non-parametric)
  - Rolling window methodology
  - Why VaR is negative of quantile (losses are negative returns)

### Component: src/models_monte_carlo.py (Refactored)
- **AI Tool:** Claude Code
- **Prompt:** "Refactor Monte Carlo VaR model to function-based approach with docstrings and config usage"
- **Generated Code:** Complete refactored models_monte_carlo.py (~147 lines)
- **Modifications:** None - used as generated
- **Understanding:** Yes, I understand:
  - Parametric simulation assuming normal distribution
  - Uses config.RANDOM_SEED for reproducibility
  - Handles edge case of zero volatility
  - 100,000 simulations per forecast

### Key Takeaway from Phase 3
**THE MOST IMPORTANT FIX:** The VIX model now uses lagged VIX for true forecasting. This was the critical methodological flaw in the original implementation. Without this fix, the results would have been meaningless due to look-ahead bias.

---

## Summary (To be updated as project progresses)

- **Total Components:** 8
- **Lines Generated:** ~800
- **Lines Modified:** ~15
- **AI Assistance Percentage:** ~98%
- **Understanding Level:** Complete understanding of all generated code, especially the critical VIX lag fix
