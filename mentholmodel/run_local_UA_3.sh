#!/bin/bash

dateime=2025-01-27_20-31-10-293992

echo $dateime


# python -m uncertainty_analysis_do_runs 3 3 3 0 "$datetime"
# python -m uncertainty_analysis_do_runs 3 3 3 1 "$datetime"
# python -m uncertainty_analysis_do_runs 3 3 3 2 "$datetime"
# python -m uncertainty_analysis_do_runs 3 3 3 3 "$datetime"
# python -m uncertainty_analysis_do_runs 3 3 3 4 "$datetime"
python -m uncertainty_analysis_do_runs 3 3 3 5 2025-01-27_20-31-10-293992 &
sleep 1
python -m uncertainty_analysis_do_runs 3 3 3 5 2025-01-27_20-31-10-293992 --second_half

