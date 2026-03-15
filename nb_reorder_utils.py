
import re

def reorder_py_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Define segments based on cell delimiters '# %%'
    cells = []
    current_cell = []
    for line in lines:
        if line.startswith('# %%'):
            if current_cell:
                cells.append(current_cell)
            current_cell = [line]
        else:
            current_cell.append(line)
    if current_cell:
        cells.append(current_cell)

    print(f"Total cells found: {len(cells)}")

    # Identify cells
    new_cells = []
    misplaced_indices = []
    dim_red_cells = []
    strategic_cells = []
    
    # Standard cells in order
    flow_ordered = []

    for i, cell in enumerate(cells):
        content = "".join(cell)
        
        # Check for misplaced silhouette
        if "### 5.1 Quantitative Validation: Silhouette Analysis" in content or "# Silhouette Analysis for K-Means Validation" in content:
            if i < 20: # It's only misplaced if it's early
                print(f"Cell {i} is misplaced silhouette. Skipping for now.")
                misplaced_indices.append(i)
                continue
        
        # Check for original Section 8 (Strategic)
        if "# 8. Strategic Insights & Conclusion" in content or "# 8.1 Which segment is most valuable?" in content or "# 8.2 Deployment: Exporting the Results" in content or "avg_spending =" in content or "df_final =" in content:
             strategic_cells.append(cell)
             continue
        
        # Check for new Section 8 (Dimensionality Reduction)
        if "# 8. Dimensionality Reduction & Advanced Visualization" in content or "# 8.1 Global Structure vs. Local Manifold" in content or "# PCA: 2D Projection" in content:
            dim_red_cells.append(cell)
            continue
            
        flow_ordered.append(cell)

    # Now assemble
    # 1. Flow up to Section 7
    # 2. Add Dimensionality Reduction (Renumbered to 8)
    # 3. Add Strategic Insights (Renumbered to 9)

    final_cells = []
    for cell in flow_ordered:
        final_cells.append(cell)
        
        # After DBSCAN (Section 7), insert DimRed then Strategic
        content = "".join(cell)
        if "# 7. Advanced Anomaly Detection (DBSCAN)" in content:
            # Check if this is the header cell, if so, keep going until we find the end of Section 7
            pass
        
        # Let's find the last DBSCAN cell
        if i == len(flow_ordered) - 1: # We'll append at the end of flow_ordered for now
            pass

    # Actually, let's just build the sequence manually to be safe
    final_output_cells = []
    
    # 1. Cleaned Flow (everything except Strategic and DimRed and Misplaced)
    final_output_cells.extend(flow_ordered)
    
    # 2. Add DimRed
    # Renumber DimRed cells
    for cell in dim_red_cells:
        content = "".join(cell)
        content = content.replace("# 8. ", "# 8. ") # Keep as 8
        content = content.replace("# 8.1 ", "# 8.1 ")
        # Fix the model.labels_ bug
        content = content.replace("c=model.labels_", "c=df['Cluster']")
        # Update theme if easy? No, viridis is okay for now, but let's try to be consistent
        final_output_cells.append([content])

    # 3. Add Strategic
    for cell in strategic_cells:
        content = "".join(cell)
        content = content.replace("# 8. ", "# 9. ")
        content = content.replace("# 8.1 ", "# 9.1 ")
        content = content.replace("# 8.2 ", "# 9.2 ")
        final_output_cells.append([content])

    # 4. Improvements: Add numerical silhouette score to K-Means
    # Look for SilhouetteVisualizer in K-Means section
    for i, cell in enumerate(final_output_cells):
        content = "".join(cell)
        if "visualizer = SilhouetteVisualizer(" in content and "model_5" in content:
            # Add print score after show()
            if "visualizer.show()" in content:
                new_content = content.replace("visualizer.show()", 
                                             "visualizer.show()\n\n# Numeric Silhouette Score\nfrom sklearn.metrics import silhouette_score\nscore = silhouette_score(X_scaled, model_5.labels_)\nprint(f'Average Silhouette Score for K=5: {score:.4f}')")
                final_output_cells[i] = [new_content]

        # Add score to DBSCAN
        if "plt.title('DBSCAN Clustering:" in content:
             if "plt.show()" in content:
                 replacement = "plt.show()\n\n# Silhouette Score for DBSCAN (excluding noise)\nfrom sklearn.metrics import silhouette_score\nif len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0) > 1:\n    score_dbscan = silhouette_score(X_scaled[clusters_dbscan != -1], clusters_dbscan[clusters_dbscan != -1])\n    print(f'DBSCAN Silhouette Score (without noise): {score_dbscan:.4f}')\nelse:\n    print('DBSCAN: Not enough clusters for silhouette score calculation.')"
                 new_content = content.replace("plt.show()", replacement)
                 final_output_cells[i] = [new_content]

    # Write back
    with open(output_path, 'w', encoding='utf-8') as f:
        for cell in final_output_cells:
            f.write("".join(cell))
            if not "".join(cell).endswith('\n'):
                f.write('\n')

if __name__ == "__main__":
    reorder_py_file('/home/leondgarse/workspace/msi5102_group_project/customer-segmentation-eda-k-means-dbscan.py', 
                   '/home/leondgarse/workspace/msi5102_group_project/customer-segmentation-eda-k-means-dbscan_refined.py')
