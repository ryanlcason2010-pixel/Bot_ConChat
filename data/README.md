# Framework Database Schema

This directory should contain the `frameworks.xlsx` file with your consulting frameworks.

## Required Columns

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| id | Integer | Yes | Unique identifier for the framework |
| name | String | Yes | Name of the framework (e.g., "SPIN Selling") |
| type | String | Yes | Category type (e.g., "Sales", "Strategy", "Operations") |
| business_domains | String | Yes | Comma-separated domains (e.g., "B2B Sales, Enterprise") |
| problem_symptoms | String | Yes | Problems this framework addresses |
| use_case | String | Yes | When to use this framework |

## Optional Columns

| Column | Type | Default | Description |
|--------|------|---------|-------------|
| inputs_required | String | "" | What data/inputs are needed |
| outputs_artifacts | String | "" | What the framework produces |
| diagnostic_questions | String | "" | Pipe-separated questions for diagnosis |
| red_flag_indicators | String | "" | Warning signs to look for |
| levers | String | "" | Key levers/actions to pull |
| tags | String | "" | Comma-separated tags for search |
| difficulty_level | String | "intermediate" | beginner/intermediate/advanced |
| related_frameworks | String | "" | Comma-separated related framework IDs |

## Example Row

```
id: 1
name: SPIN Selling
type: Sales
business_domains: B2B Sales, Enterprise Sales, Complex Sales
problem_symptoms: Long sales cycles, Low win rates, Deals stalling
use_case: Use when selling complex B2B solutions with multiple stakeholders
inputs_required: Customer conversation access, Sales opportunity data
outputs_artifacts: Qualification criteria, Question framework, Deal strategy
diagnostic_questions: How long is your typical sales cycle?|What percentage of deals do you win?|Where do deals typically stall?
red_flag_indicators: Sales reps skip discovery phase|Focus on features not outcomes|No documented customer pain points
levers: Deep discovery questions|Implication development|Need-payoff connections
tags: sales, discovery, qualification, b2b
difficulty_level: intermediate
related_frameworks: 2,5,12
```

## Notes

- Pipe (`|`) is used to separate diagnostic questions
- Commas are used to separate tags, domains, and related framework IDs
- The file should be saved as `.xlsx` format (Excel 2007+)
- UTF-8 encoding is recommended for special characters
