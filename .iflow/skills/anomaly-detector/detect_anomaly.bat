@echo off
REM Anomaly Detector - Windows Batch Script
REM Detect anomalies in time series data

python "%~dp0scripts\detect_anomaly.py" %*
