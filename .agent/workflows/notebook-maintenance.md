---
description: How to maintain and update the customer segmentation notebook
---

# Notebook Maintenance Workflow

This workflow describes the process for updating the `customer-segmentation-eda-k-means-dbscan.ipynb` notebook. To ensure clean formatting and structural integrity, we edit the Python script version and then sync changes back to the notebook.

## Steps

### 1. Edit the Python Script
Modify the core logic or documentation in the Python script:
`customer-segmentation-eda-k-means-dbscan.py`

### 2. Clean & Normalize (Optional)
If you made significant changes to markdown blocks or added HTML styles, run the cleanup utility:
// turbo
```bash
python3 nb_clean_utils.py
```

### 3. Convert Script to Notebook
Synchronize the changes from the `.py` script back to the `.ipynb` file:
// turbo
```bash
python3 nb_convert_utils.py
```

### 4. Verify Structure
Audit the generated notebook to ensure cell numbering and segments are correct:
// turbo
```bash
python3 nb_map_utils.py
```

### 5. Final Execution Check
Open the generated `.ipynb` in your preferred editor (VS Code, Jupyter, or Colab) and run all cells to verify that the visualizations and mathematical outputs are consistent with the graduate-level requirements.
