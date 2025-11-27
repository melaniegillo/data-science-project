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

## Summary (To be updated as project progresses)

- **Total Components:** 3
- **Lines Generated:** ~90
- **Lines Modified:** ~5
- **AI Assistance Percentage:** ~95%
- **Understanding Level:** Complete understanding of all generated code
