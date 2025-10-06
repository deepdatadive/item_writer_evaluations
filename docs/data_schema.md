# Data Schema Documentation

## Overview

This document describes the standard data schema for experiments in this repository.

## Long Format Data Structure

All experimental data should be stored in **long format** (also called "tidy" format), where:
- Each row represents a single observation (one participant's response to one item)
- Each column represents a variable
- Each cell contains a single value

### Standard Columns

#### Required Columns

| Column Name | Type | Description | Example Values |
|------------|------|-------------|----------------|
| `participant_id` | string/int | Unique identifier for each participant | "P001", "P002" |
| `item_id` | string/int | Unique identifier for each item/question | "Q001", "item_1" |
| `response` | varies | Participant's raw response | "A", "correct", 42 |

#### Recommended Columns

| Column Name | Type | Description | Example Values |
|------------|------|-------------|----------------|
| `condition` | string | Experimental condition | "control", "ai_generated" |
| `correct` | int | Whether response was correct (0/1) | 0, 1 |
| `response_time` | float | Time taken to respond (seconds) | 12.5, 45.2 |
| `item_type` | string | Type or category of item | "multiple_choice", "open_ended" |
| `difficulty` | string/float | Difficulty level | "easy", "medium", "hard", or 0.0-1.0 |
| `timestamp` | datetime | When response was recorded | "2025-10-06 14:30:00" |

#### Optional Participant-Level Columns

| Column Name | Type | Description | Example Values |
|------------|------|-------------|----------------|
| `age` | int | Participant age | 25, 34 |
| `gender` | string | Gender | "M", "F", "Other" |
| `education` | string | Education level | "High School", "Bachelor's", "PhD" |
| `session_id` | string | Testing session identifier | "session_001" |

#### Optional Item-Level Columns

| Column Name | Type | Description | Example Values |
|------------|------|-------------|----------------|
| `ai_model` | string | AI model used to generate item | "gpt-4", "claude-3" |
| `prompt_version` | string | Prompt version used | "v1", "v2" |
| `item_source` | string | Source of the item | "ai_generated", "human_written" |
| `content_domain` | string | Subject area | "math", "reading" |
| `bloom_level` | string | Cognitive level | "knowledge", "comprehension" |

### Example Data

```csv
participant_id,condition,item_id,item_type,response,correct,response_time,difficulty
P001,control,Q001,multiple_choice,A,1,12.5,easy
P001,control,Q002,multiple_choice,C,0,18.3,medium
P001,control,Q003,multiple_choice,B,1,25.1,hard
P002,treatment,Q001,multiple_choice,A,1,10.2,easy
P002,treatment,Q002,multiple_choice,B,1,15.8,medium
P002,treatment,Q003,multiple_choice,B,1,20.5,hard
```

## File Naming Conventions

### Raw Data Files

Format: `responses_YYYYMMDD.csv` or `responses_YYYYMMDD_description.csv`

Examples:
- `responses_20251006.csv`
- `responses_20251006_pilot.csv`
- `responses_20251010_main_study.csv`

### Processed Data Files

Format: `[description]_[status].csv`

Examples:
- `cleaned_responses.csv`
- `aggregated_by_participant.csv`
- `item_statistics.csv`

## Data Quality Checks

All data should pass these validation checks:

1. **No duplicate responses**: Each (participant_id, item_id) pair should appear only once
2. **No missing values**: Required columns should have no null values
3. **Valid values**: Categorical variables should only contain expected values
4. **Consistent IDs**: All IDs should be consistently formatted
5. **Logical relationships**: e.g., if `correct` is 1, the response should match the correct answer

## Metadata Files

Each experiment should include a `metadata.json` file describing:
- Experiment design
- Variable definitions
- Data collection dates
- Any deviations from standard schema
- Notes about data quality issues

See `experiments/exp001_baseline_comparison/data/metadata.json` for a template.
