"""Worker modules for the CV-as-Contract pipeline.

Imported by main.py at the repo root. Each module owns one pipeline stage:

    scheme_analysis     — CV_PLAN author (frozen contract)
    cv_engine           — CVEngine + OOF aggregator
    feature_engineer    — fold-bound transformer factory
    modeler             — CV-agnostic per-fold estimator adapter
    critic              — read-only CV consistency auditor
    report_writer       — CV-aware report generator
"""
