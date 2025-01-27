#!/bin/bash

# Check if the correct number of arguments is provided
if [ $# -lt 1 ]; then
    echo "Error: Missing arguments."
    echo "Usage: $0 <num of mortsets>"
    exit 1
fi

# Execute the first command and capture its output
echo ""
echo "Creating UA params"
output=$(python -m uncertainty_analysis_create_params $1 $1 $1)

echo $output

# Use regex to extract the datetime string from the output
if [[ $output =~ ([0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{6}) ]]; then
    datetime="${BASH_REMATCH[1]}"
    echo ""

    # Use the extracted datetime as an argument to a secondary command
    python -m uncertainty_analysis_do_runs $1 $1 $1 0 "$datetime" &
    sleep 1
    python -m uncertainty_analysis_do_runs $1 $1 $1 0 "$datetime" --second_half

    python -m uncertainty_analysis_do_runs $1 $1 $1 1 "$datetime" &
    sleep 1
    python -m uncertainty_analysis_do_runs $1 $1 $1 1 "$datetime" --second_half

    python -m uncertainty_analysis_do_runs $1 $1 $1 2 "$datetime" &
    sleep 1
    python -m uncertainty_analysis_do_runs $1 $1 $1 2 "$datetime" --second_half

    python -m uncertainty_analysis_do_runs $1 $1 $1 3 "$datetime" &
    sleep 1
    python -m uncertainty_analysis_do_runs $1 $1 $1 3 "$datetime" --second_half

    python -m uncertainty_analysis_do_runs $1 $1 $1 4 "$datetime" &
    sleep 1
    python -m uncertainty_analysis_do_runs $1 $1 $1 4 "$datetime" --second_half

    python -m uncertainty_analysis_do_runs $1 $1 $1 5 "$datetime" &
    sleep 1
    python -m uncertainty_analysis_do_runs $1 $1 $1 5 "$datetime" --second_half
else
    echo "No datetime string found in the output."
    exit 1
fi