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

## Summary (To be updated as project progresses)

- **Total Components:** 5
- **Lines Generated:** ~350
- **Lines Modified:** ~10
- **AI Assistance Percentage:** ~97%
- **Understanding Level:** Complete understanding of all generated code
