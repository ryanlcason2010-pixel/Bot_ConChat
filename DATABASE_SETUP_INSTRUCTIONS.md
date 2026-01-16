# 🗄️ Framework Database - Complete Setup Guide

## ✅ What You Have

**One complete SQL script** that creates your entire framework database:
- **File:** `complete_database_setup.sql` (648 KB)
- **Frameworks:** 852 diagnostic frameworks
- **Business Functions:** 7 categories
- **Database:** SQLite (single file, zero config)

---

## 🚀 Quick Setup (3 Methods)

### Method 1: Command Line (Easiest)

```bash
# Create database from SQL script
sqlite3 frameworks.db < complete_database_setup.sql

# Verify it worked
sqlite3 frameworks.db "SELECT COUNT(*) FROM frameworks;"
# Should output: 852

# Done! You now have frameworks.db ready to use
```

### Method 2: Python Script

```python
import sqlite3

# Create database from SQL script
conn = sqlite3.connect('frameworks.db')

with open('complete_database_setup.sql', 'r', encoding='utf-8') as f:
    sql_script = f.read()
    conn.executescript(sql_script)

conn.close()

print("✓ Database created: frameworks.db")
```

### Method 3: Interactive SQL

```bash
# Start SQLite
sqlite3 frameworks.db

# Inside SQLite prompt:
.read complete_database_setup.sql

# Verify:
SELECT COUNT(*) FROM frameworks;
-- Output: 852

.quit
```

---

## 📊 Database Schema

### Main Table: `frameworks`

| Column | Type | Description |
|--------|------|-------------|
| **id** | INTEGER PRIMARY KEY | Auto-incrementing ID |
| **framework_name** | TEXT | Name (e.g., "SPIN Selling") |
| **business_function** | TEXT | Category (Sales, Finance, etc.) |
| **framework_type** | TEXT | "diagnostic" or "learning" |
| **sub_category** | TEXT | Detailed categorization |
| **lifecycle_stages** | TEXT | When to use in business lifecycle |
| **skills_required** | TEXT | Skills needed to implement |
| **diagnostic_questions** | TEXT | Questions to ask clients |
| **red_flag_indicators** | TEXT | Warning signs to look for |
| **levers** | TEXT | Controllable variables/actions |
| **current_state_assessment** | TEXT | Assessment notes |
| **priority_level** | TEXT | Priority classification |
| **related_canon** | TEXT | Related frameworks |
| **notes** | TEXT | Additional notes |
| **created_at** | TIMESTAMP | Record creation time |
| **updated_at** | TIMESTAMP | Last update time |

### Full-Text Search Table: `frameworks_fts`

Virtual FTS5 table for fast text search across:
- Framework names
- Business functions
- Diagnostic questions
- Red flags
- Levers

### Indexes

Optimized for fast queries on:
- `framework_name`
- `business_function`
- `framework_type`
- `sub_category`
- `priority_level`

---

## 📈 Data Distribution

| Business Function | Count |
|------------------|-------|
| Finance & Economics | 169 |
| Direct Response Marketing | 154 |
| People & Talent | 153 |
| Technology & Systems | 138 |
| Operations & Delivery | 111 |
| Sales & Business Development | 67 |
| Branding & Positioning | 60 |
| **TOTAL** | **852** |

---

## 💻 Usage Examples

### Python

```python
import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('frameworks.db')

# Example 1: Get all Sales frameworks
query = """
    SELECT framework_name, sub_category, diagnostic_questions
    FROM frameworks
    WHERE business_function = 'Sales & Business Development'
"""
sales_df = pd.read_sql_query(query, conn)
print(sales_df)

# Example 2: Search for "retention" frameworks
query = """
    SELECT framework_name, business_function
    FROM frameworks
    WHERE framework_name LIKE '%retention%'
       OR diagnostic_questions LIKE '%retention%'
"""
retention_df = pd.read_sql_query(query, conn)

# Example 3: Full-text search
query = """
    SELECT f.framework_name, f.business_function
    FROM frameworks f
    JOIN frameworks_fts ON f.id = frameworks_fts.rowid
    WHERE frameworks_fts MATCH 'employee turnover'
"""
results_df = pd.read_sql_query(query, conn)

conn.close()
```

### SQL (Direct Queries)

```sql
-- Count by business function
SELECT business_function, COUNT(*) as count
FROM frameworks
GROUP BY business_function
ORDER BY count DESC;

-- Find frameworks by sub-category
SELECT framework_name, business_function
FROM frameworks
WHERE sub_category LIKE '%Retention%';

-- Get frameworks with diagnostic questions
SELECT framework_name, diagnostic_questions
FROM frameworks
WHERE diagnostic_questions IS NOT NULL
  AND diagnostic_questions != ''
LIMIT 10;

-- Search across all text fields
SELECT framework_name, business_function
FROM frameworks
WHERE framework_name LIKE '%pricing%'
   OR levers LIKE '%pricing%'
   OR diagnostic_questions LIKE '%pricing%';
```

### Python - Advanced Usage

```python
import sqlite3

def search_frameworks(search_term):
    """Search for frameworks by keyword"""
    conn = sqlite3.connect('frameworks.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT framework_name, business_function, diagnostic_questions
        FROM frameworks
        WHERE framework_name LIKE ? 
           OR diagnostic_questions LIKE ?
           OR levers LIKE ?
    """, (f'%{search_term}%', f'%{search_term}%', f'%{search_term}%'))
    
    results = cursor.fetchall()
    conn.close()
    
    return results

# Usage
results = search_frameworks("high CAC")
for name, function, questions in results:
    print(f"{name} ({function})")
```

---

## 🔍 Powerful Features

### 1. Full-Text Search

```sql
-- Search with FTS5 (much faster for text search)
SELECT f.framework_name, f.business_function
FROM frameworks f
JOIN frameworks_fts ON f.id = frameworks_fts.rowid
WHERE frameworks_fts MATCH 'employee AND retention'
LIMIT 10;
```

### 2. Filter by Multiple Criteria

```python
import sqlite3

conn = sqlite3.connect('frameworks.db')

# Complex filtering
query = """
    SELECT framework_name, sub_category, diagnostic_questions
    FROM frameworks
    WHERE business_function = 'People & Talent'
      AND sub_category LIKE '%Retention%'
      AND priority_level IS NOT NULL
"""

cursor = conn.cursor()
cursor.execute(query)
results = cursor.fetchall()

conn.close()
```

### 3. Export Specific Categories

```python
import pandas as pd
import sqlite3

def export_business_function(function_name, output_file):
    """Export a specific business function to CSV"""
    conn = sqlite3.connect('frameworks.db')
    
    query = """
        SELECT framework_name, sub_category, diagnostic_questions,
               red_flag_indicators, levers
        FROM frameworks
        WHERE business_function = ?
    """
    
    df = pd.read_sql_query(query, conn, params=(function_name,))
    df.to_csv(output_file, index=False)
    
    conn.close()
    print(f"✓ Exported {len(df)} frameworks to {output_file}")

# Usage
export_business_function('Finance & Economics', 'finance_frameworks.csv')
```

---

## 🔧 Integration with Framework Assistant

Replace Excel loading with database queries:

```python
# In your utils/loader.py

import sqlite3
import pandas as pd

def load_frameworks(db_path='frameworks.db'):
    """Load frameworks from database instead of Excel"""
    
    conn = sqlite3.connect(db_path)
    
    # Load all frameworks
    df = pd.read_sql_query("""
        SELECT 
            id,
            framework_name as name,
            business_function as business_domains,
            framework_type as type,
            sub_category,
            diagnostic_questions as problem_symptoms,
            levers as use_case,
            diagnostic_questions,
            red_flag_indicators,
            levers,
            related_canon
        FROM frameworks
    """, conn)
    
    # Create searchable_text for embeddings
    df['searchable_text'] = (
        'Framework: ' + df['name'] + ' | ' +
        'Domain: ' + df['business_domains'] + ' | ' +
        'Problem: ' + df['problem_symptoms'].fillna('') + ' | ' +
        'Levers: ' + df['levers'].fillna('')
    )
    
    conn.close()
    
    return df

# Usage
df = load_frameworks()
print(f"Loaded {len(df)} frameworks")
```

**Benefits:**
- ✅ 5-10x faster than Excel loading
- ✅ No Excel file dependency
- ✅ Easy to query/filter before loading
- ✅ Can update frameworks without Excel
- ✅ Production-ready

---

## 🛠️ Database Maintenance

### Add New Framework

```python
import sqlite3

conn = sqlite3.connect('frameworks.db')
cursor = conn.cursor()

cursor.execute("""
    INSERT INTO frameworks (
        framework_name, business_function, framework_type,
        sub_category, diagnostic_questions, levers
    ) VALUES (?, ?, ?, ?, ?, ?)
""", (
    "New Framework Name",
    "Sales & Business Development",
    "diagnostic",
    "Custom Category",
    "Question 1|Question 2|Question 3",
    "Lever 1 • Lever 2 • Lever 3"
))

conn.commit()
conn.close()
```

### Update Framework

```python
cursor.execute("""
    UPDATE frameworks
    SET diagnostic_questions = ?,
        updated_at = CURRENT_TIMESTAMP
    WHERE framework_name = ?
""", (
    "Updated questions here",
    "Market Segmentation & ICP"
))

conn.commit()
```

### Backup Database

```bash
# Simple copy
cp frameworks.db frameworks_backup_$(date +%Y%m%d).db

# Or SQLite backup command
sqlite3 frameworks.db ".backup frameworks_backup.db"
```

---

## 📊 Verification Queries

After setting up, verify everything worked:

```sql
-- Total frameworks
SELECT COUNT(*) FROM frameworks;
-- Expected: 852

-- Count by business function
SELECT business_function, COUNT(*) as count
FROM frameworks
GROUP BY business_function
ORDER BY count DESC;

-- Check FTS index
SELECT COUNT(*) FROM frameworks_fts;
-- Expected: 852

-- Sample data
SELECT id, framework_name, business_function
FROM frameworks
LIMIT 5;

-- Check for missing data
SELECT 
    COUNT(CASE WHEN diagnostic_questions = '' THEN 1 END) as missing_questions,
    COUNT(CASE WHEN levers = '' THEN 1 END) as missing_levers,
    COUNT(CASE WHEN red_flag_indicators = '' THEN 1 END) as missing_red_flags
FROM frameworks;
```

---

## 🆘 Troubleshooting

### "Database is locked"

```python
# Add timeout when connecting
conn = sqlite3.connect('frameworks.db', timeout=10.0)
```

### "No such table: frameworks"

```bash
# Verify the SQL script ran completely
sqlite3 frameworks.db ".schema frameworks"
# Should show the table definition
```

### "Syntax error near..."

```bash
# Make sure you're using SQLite3 (not older SQLite2)
sqlite3 --version
# Should be 3.x.x
```

### Performance is slow

```sql
-- Rebuild indexes
REINDEX;

-- Vacuum database (optimize file size)
VACUUM;

-- Analyze for query optimization
ANALYZE;
```

---

## 🔄 Migration to PostgreSQL (Future)

If you need PostgreSQL later:

```bash
# Install pgloader
pip install pgloader

# Convert SQLite to PostgreSQL
createdb frameworks_db
pgloader frameworks.db postgresql:///frameworks_db
```

Or use Python:

```python
import sqlite3
import pandas as pd
from sqlalchemy import create_engine

# Read from SQLite
sqlite_conn = sqlite3.connect('frameworks.db')
df = pd.read_sql_query("SELECT * FROM frameworks", sqlite_conn)
sqlite_conn.close()

# Write to PostgreSQL
pg_engine = create_engine('postgresql://user:pass@localhost/frameworks_db')
df.to_sql('frameworks', pg_engine, if_exists='replace', index=False)
```

---

## 📚 Additional Resources

- **SQLite Documentation:** https://www.sqlite.org/docs.html
- **Python sqlite3 Module:** https://docs.python.org/3/library/sqlite3.html
- **FTS5 Full-Text Search:** https://www.sqlite.org/fts5.html
- **Pandas SQL:** https://pandas.pydata.org/docs/reference/api/pandas.read_sql_query.html

---

## ✅ Checklist

Before using in production:

- [ ] SQL script runs without errors
- [ ] 852 frameworks imported
- [ ] All 7 business functions present
- [ ] Full-text search index created
- [ ] Sample queries return expected results
- [ ] Database backed up
- [ ] Integrated with application (if applicable)

---

## 🎉 You're Ready!

Your database is set up and ready to use. The single SQL script contains everything you need.

**Next steps:**
1. Run the setup command
2. Test with sample queries
3. Integrate with your Framework Assistant app
4. Start querying your 852 frameworks!

---

**Questions?** The SQL script is self-contained and includes schema, indexes, full-text search, and all 852 frameworks in one file.
