@echo off
setlocal

:: All norm types to test
set OPTIONS=l2 linf snr fletcher_munson leakage min_max_freqs

:: Loop over each norm type
for %%N in (%OPTIONS%) do (
    python main.py --small_data --norm_type %%N
)

endlocal
