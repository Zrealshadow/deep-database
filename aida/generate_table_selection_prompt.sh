#!/bin/bash

# Script to generate all table selection prompts and save them in static/table_selection directory
# Usage: ./generate_table_selection_prompt.sh

set -e  # Exit on any error

# Configuration
OUTPUT_DIR="static/table_selection"
PROMPT_TYPE="table_selection"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "==================================================================="
echo "Generating Table Selection Prompts"
echo "==================================================================="
echo "Output directory: $OUTPUT_DIR"
echo "Prompt type: $PROMPT_TYPE"
echo ""

# Define database and task combinations
declare -A DB_TASKS=(
    ["event"]="user-attendance user-ignore user-repeat"
    ["stack"]="post-votes user-badge user-engagement"
    ["avito"]="ad-ctr user-click user-visit"
    ["trial"]="site-success study-adverse study-outcome"
    ["ratebeer"]="user-active place-positive beer-positive"
    ["f1"]="driver-dnf"
    ["amazon"]="user-churn product-rating"
)

# Function to generate prompt for a specific database/task combination
generate_prompt() {
    local db_name="$1"
    local task_name="$2"
    local output_file="$3"

    # Check if file already exists
    if [ -f "$output_file" ]; then
        echo "⏭️  Skipping $db_name/$task_name (file already exists): $output_file"
        return 0
    fi

    echo "Generating prompt for $db_name/$task_name..."

    # Run the AIDA prompt generation command
    python -m aida.cmd.print_prompt "$db_name" "$task_name" "$PROMPT_TYPE" --output "$output_file"

    if [ $? -eq 0 ]; then
        echo "✅ Successfully generated: $output_file"
    else
        echo "❌ Failed to generate prompt for $db_name/$task_name"
        return 1
    fi
}

# Counter for tracking progress
total_prompts=0
generated_prompts=0
skipped_prompts=0
failed_prompts=0

# Calculate total number of prompts to generate
for db_name in "${!DB_TASKS[@]}"; do
    tasks=(${DB_TASKS[$db_name]})
    total_prompts=$((total_prompts + ${#tasks[@]}))
done

echo "Total prompts to generate: $total_prompts"
echo ""

# Generate prompts for each database/task combination
for db_name in "${!DB_TASKS[@]}"; do
    echo "Processing database: $db_name"
    echo "-------------------------------------------------------------------"

    # Create database-specific subdirectory
    db_output_dir="$OUTPUT_DIR/$db_name"
    mkdir -p "$db_output_dir"

    # Split tasks string into array
    tasks=(${DB_TASKS[$db_name]})

    for task_name in "${tasks[@]}"; do
        # Create output filename
        output_file="$db_output_dir/${task_name}_table_selection.txt"

        # Check if file exists before calling generate_prompt
        if [ -f "$output_file" ]; then
            generate_prompt "$db_name" "$task_name" "$output_file"  # This will skip
            skipped_prompts=$((skipped_prompts + 1))
        else
            # File doesn't exist, attempt to generate
            if generate_prompt "$db_name" "$task_name" "$output_file"; then
                generated_prompts=$((generated_prompts + 1))
            else
                failed_prompts=$((failed_prompts + 1))
            fi
        fi

        echo ""
    done

    echo ""
done

# Generate summary report
echo "==================================================================="
echo "Generation Summary"
echo "==================================================================="
echo "Total prompts: $total_prompts"
echo "Successfully generated: $generated_prompts"
echo "Skipped (already exist): $skipped_prompts"
echo "Failed: $failed_prompts"
echo ""

if [ $failed_prompts -eq 0 ]; then
    if [ $skipped_prompts -eq 0 ]; then
        echo "🎉 All prompts generated successfully!"
    else
        echo "✅ All prompts processed successfully! ($generated_prompts new, $skipped_prompts skipped)"
    fi
else
    echo "⚠️  Some prompts failed to generate. Check the output above for details."
fi

# Create index file with all generated prompts
index_file="$OUTPUT_DIR/index.txt"
echo "Creating index file: $index_file"

cat > "$index_file" << EOF
Table Selection Prompts Index
============================
Generated on: $(date)
Total prompts: $generated_prompts

Directory Structure:
EOF

# Add directory tree to index
find "$OUTPUT_DIR" -name "*.txt" -not -name "index.txt" | sort | while read -r file; do
    # Get relative path from output directory
    rel_path="${file#$OUTPUT_DIR/}"
    echo "  $rel_path" >> "$index_file"
done

cat >> "$index_file" << EOF

Usage:
------
Each file contains a complete table selection prompt for the specified database and task.
Files are organized by database in subdirectories.

Naming Convention:
- {database_name}/{task_name}_table_selection.txt

Example Files:
- event/user-attendance_table_selection.txt
- stack/user-badge_table_selection.txt
- ratebeer/user-active_table_selection.txt
EOF

echo "✅ Index file created: $index_file"
echo ""

# Display final directory structure
echo "Final directory structure:"
echo "-------------------------------------------------------------------"
find "$OUTPUT_DIR" -type f | sort

echo ""
echo "==================================================================="
echo "Script completed!"
echo "==================================================================="