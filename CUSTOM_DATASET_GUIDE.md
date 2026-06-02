# 📊 How to Use Custom Datasets

This guide explains how to use different datasets with the Cancer Health Prediction System.

---

## 🎯 Quick Start

### **Option 1: Use Built-in Dataset (Easiest)**

```bash
/usr/bin/python3 train_model.py
```

Uses the Breast Cancer Wisconsin dataset from scikit-learn.

---

### **Option 2: Use CSV File**

#### Step 1: Prepare Your CSV File

Your CSV file should have:
- **Rows**: Each row = one sample
- **Columns**: Features + one target column
- **Format**: Comma-separated values

**Example structure:**
```
feature1,feature2,feature3,...,target
14.5,25.3,97.8,...,0
13.2,18.4,85.3,...,1
...
```

#### Step 2: Place File in Project Directory

```bash
# Copy your CSV file to the project folder
cp /path/to/your/data.csv /home/ubuntu/Cancer_Prediction_Project/
```

#### Step 3: Run Training Script

```bash
cd /home/ubuntu/Cancer_Prediction_Project
/usr/bin/python3 train_custom_dataset.py
```

When prompted:
```
SELECT DATASET SOURCE:
1. Built-in Breast Cancer Wisconsin Dataset
2. Load CSV file from disk
3. Load Excel file from disk

Enter choice (1, 2, or 3): 2
Enter path to CSV file: data.csv
Enter target column name (press Enter to use last column): target
```

---

### **Option 3: Use Excel File**

#### Step 1: Prepare Your Excel File

- **File formats**: .xlsx or .xls
- **Structure**: Same as CSV (features + target column)

#### Step 2: Place File in Project Directory

```bash
cp /path/to/your/data.xlsx /home/ubuntu/Cancer_Prediction_Project/
```

#### Step 3: Run Training Script

```bash
/usr/bin/python3 train_custom_dataset.py
```

When prompted:
```
SELECT DATASET SOURCE:
Enter choice (1, 2, or 3): 3
Enter path to Excel file: data.xlsx
Enter target column name (press Enter to use last column): target
```

---

## 📋 Dataset Format Requirements

### Minimum Requirements

| Requirement | Details |
|-------------|---------|
| **Rows** | At least 100 samples (more is better) |
| **Columns** | At least 2 (features + target) |
| **Data Type** | Numeric values (integers or decimals) |
| **Missing Values** | Will be filled automatically with mean |
| **Target Column** | Numeric (0, 1, 2...) or text (will be auto-encoded) |

### Example CSV Format

```csv
radius,texture,perimeter,smoothness,compactness,target
13.5,18.2,85.5,0.095,0.08,0
14.2,19.5,92.3,0.098,0.09,1
12.1,16.8,78.2,0.092,0.07,0
15.3,22.1,102.1,0.105,0.12,1
```

---

## 🔄 Workflow: Using Custom Dataset

### Step-by-Step Example

**Scenario**: You have a diabetes dataset in `diabetes_data.csv`

```bash
# 1. Navigate to project
cd /home/ubuntu/Cancer_Prediction_Project

# 2. Copy your CSV file
cp ~/Downloads/diabetes_data.csv .

# 3. Run custom dataset training
/usr/bin/python3 train_custom_dataset.py

# 4. Select option 2 (CSV)
# 5. Enter filename: diabetes_data.csv
# 6. Enter target column name: diabetes_status

# 7. Wait for training to complete
# 8. Use predictions as usual
/usr/bin/python3 predict.py
```

---

## 📁 File Structure After Loading Custom Dataset

```
Cancer_Prediction_Project/
├── train_custom_dataset.py          # Custom dataset training script
├── your_custom_data.csv             # Your CSV file ← Add here
├── your_custom_data.xlsx            # Your Excel file ← Or here
├── best_cancer_model.pkl            # Generated model (updated)
├── correlation_heatmap.png          # Updated visualization
├── class_distribution.png           # Updated visualization
└── ...
```

---

## 💾 Supported Data Formats

### CSV (.csv)
```bash
# Standard comma-separated
feature1,feature2,feature3,target
10,20,30,0
11,21,31,1

# Tab-separated (still works)
feature1	feature2	feature3	target
10	20	30	0
```

### Excel (.xlsx, .xls)
```
Supports multiple sheets
Automatically reads the first sheet (or specify sheet name)
Works with Excel formulas and formatting
```

### Other Formats (with conversion)

If you have data in other formats:

**JSON to CSV**:
```bash
# Using pandas
/usr/bin/python3 << 'EOF'
import pandas as pd
df = pd.read_json('data.json')
df.to_csv('data.csv', index=False)
EOF
```

**SQL Database**:
```bash
/usr/bin/python3 << 'EOF'
import sqlite3
import pandas as pd

conn = sqlite3.connect('database.db')
df = pd.read_sql_query("SELECT * FROM table_name", conn)
df.to_csv('data.csv', index=False)
EOF
```

---

## 🎓 Example: Create Test Dataset

Want to test with a sample dataset? Create one:

```bash
/usr/bin/python3 << 'EOF'
import pandas as pd
import numpy as np

# Create sample data (100 samples, 5 features)
np.random.seed(42)
data = {
    'feature1': np.random.uniform(10, 20, 100),
    'feature2': np.random.uniform(15, 25, 100),
    'feature3': np.random.uniform(50, 150, 100),
    'feature4': np.random.uniform(0.08, 0.12, 100),
    'feature5': np.random.uniform(0.05, 0.10, 100),
    'target': np.random.choice([0, 1], 100)  # Binary classification
}

df = pd.DataFrame(data)
df.to_csv('sample_data.csv', index=False)
print("✓ Created sample_data.csv")
print(df.head())
EOF
```

Then train with:
```bash
/usr/bin/python3 train_custom_dataset.py
# Select option 2 (CSV)
# Enter filename: sample_data.csv
# Enter target column: target
```

---

## ⚠️ Common Issues & Solutions

### Issue: "File not found"
**Solution**: 
- Check file is in the project directory
- Use full path: `/home/ubuntu/Cancer_Prediction_Project/data.csv`
- Check filename spelling (case-sensitive)

```bash
# List files to verify
ls -la /home/ubuntu/Cancer_Prediction_Project/
```

### Issue: "Column not found"
**Solution**: 
- Check column name spelling exactly
- View columns in your file:

```bash
/usr/bin/python3 << 'EOF'
import pandas as pd
df = pd.read_csv('your_file.csv')
print(df.columns.tolist())
EOF
```

### Issue: "Data type error"
**Solution**: 
- Ensure all values are numeric
- Convert text columns manually:

```bash
/usr/bin/python3 << 'EOF'
import pandas as pd

df = pd.read_csv('data.csv')
# Convert text to numbers
df['status'] = df['status'].map({'healthy': 0, 'sick': 1})
df.to_csv('data_cleaned.csv', index=False)
print("✓ Data cleaned and saved")
EOF
```

### Issue: "Too few samples"
**Solution**: 
- Need at least 100 samples
- Use multiple files combined:

```bash
/usr/bin/python3 << 'EOF'
import pandas as pd

# Combine multiple CSV files
df1 = pd.read_csv('data1.csv')
df2 = pd.read_csv('data2.csv')
df_combined = pd.concat([df1, df2], ignore_index=True)
df_combined.to_csv('combined_data.csv', index=False)
print(f"✓ Combined {len(df_combined)} samples")
EOF
```

---

## 🔍 Verify Your Dataset

Before training, verify your dataset:

```bash
/usr/bin/python3 << 'EOF'
import pandas as pd
import numpy as np

# Load your CSV
df = pd.read_csv('your_file.csv')

print("Dataset Information:")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nStatistics:\n{df.describe()}")

# Check for non-numeric columns
non_numeric = df.select_dtypes(exclude=[np.number]).columns
if len(non_numeric) > 0:
    print(f"\n⚠️ Non-numeric columns (need encoding): {non_numeric.tolist()}")
EOF
```

---

## 🚀 Popular Public Datasets

You can download datasets from:

1. **Kaggle** (https://www.kaggle.com/datasets)
   - Cancer datasets, medical data, classification datasets
   - Download as CSV

2. **UCI Machine Learning Repository** (https://archive.ics.uci.edu/)
   - 500+ datasets
   - Various formats

3. **Google Dataset Search** (https://datasetsearch.research.google.com/)
   - Search academic datasets

4. **GitHub** (Search "dataset csv")
   - Free public datasets

### Quick Download Example

```bash
# Example: Download a dataset from GitHub
wget https://raw.githubusercontent.com/uciml/datasets/master/breast_cancer.csv
/usr/bin/python3 train_custom_dataset.py
```

---

## 📊 Comparing Models on Different Datasets

Once you train on a new dataset, compare results:

```bash
# Train on new dataset
/usr/bin/python3 train_custom_dataset.py

# Test predictions
/usr/bin/python3 predict.py

# Check web interface
/usr/bin/python3 app.py
# Visit http://localhost:5000
```

---

## 💡 Tips for Best Results

1. **Data Quality**
   - Remove duplicate rows
   - Handle missing values
   - Remove outliers if needed

2. **Feature Engineering**
   - Scale features (done automatically)
   - Remove correlated features
   - Create new meaningful features

3. **Dataset Size**
   - Minimum: 100 samples
   - Better: 500+ samples
   - Best: 1000+ samples

4. **Class Balance**
   - Try to have balanced classes
   - Unbalanced data can bias models

5. **Feature Count**
   - Minimum: 2 features
   - Optimal: 10-50 features
   - Too many can reduce accuracy

---

## 📝 Example: Complete Workflow

```bash
# 1. Create sample dataset
/usr/bin/python3 << 'EOF'
import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 200

data = pd.DataFrame({
    'age': np.random.uniform(20, 80, n_samples),
    'glucose': np.random.uniform(80, 200, n_samples),
    'blood_pressure': np.random.uniform(60, 140, n_samples),
    'bmi': np.random.uniform(18, 40, n_samples),
    'insulin': np.random.uniform(0, 300, n_samples),
    'diabetes': np.random.choice([0, 1], n_samples, p=[0.65, 0.35])
})

data.to_csv('health_data.csv', index=False)
print("✓ Created health_data.csv with 200 samples")
EOF

# 2. Train model on new dataset
/usr/bin/python3 train_custom_dataset.py
# Select: 2 (CSV)
# File: health_data.csv
# Target: diabetes

# 3. Test with CLI
/usr/bin/python3 predict.py

# 4. Test with web
/usr/bin/python3 app.py
# Open http://localhost:5000
```

---

**Need help with a specific dataset? Just ask!** 🤖

