@echo off
rem Relais LoL Coach : double-clic pour laisser ce PC créer les fiches demandées par les
rem sessions Claude du cloud (Gemini par agy, avec ton compte Google). Ferme la fenêtre pour arrêter.
title Relais LoL Coach (fiches via agy)
cd /d "%~dp0.."
if "%LOLCOACH_FICHES%"=="" set LOLCOACH_FICHES=agy
if exist ".venv\Scripts\lolcoach-relais.exe" (
  ".venv\Scripts\lolcoach-relais.exe" surveiller %*
) else (
  lolcoach-relais surveiller %*
)
pause
