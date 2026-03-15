
import nbformat
import sys

def py_to_ipynb(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by '# %%'
    cells_data = content.split('# %%')
    
    nb = nbformat.v4.new_notebook()
    cells = []

    for cell_text in cells_data:
        cell_text = cell_text.strip()
        if not cell_text:
            continue
            
        if cell_text.startswith('[markdown]'):
            # It's markdown
            # Remove '[markdown]' and find the content
            source = cell_text.replace('[markdown]', '', 1).strip()
            # The source is usually commented out with '# ' or '#'
            lines = source.split('\n')
            clean_lines = []
            for line in lines:
                if line.startswith('# '):
                    clean_lines.append(line[2:])
                elif line.startswith('#'):
                    clean_lines.append(line[1:])
                else:
                    clean_lines.append(line)
            cells.append(nbformat.v4.new_markdown_cell("\n".join(clean_lines)))
        else:
            # It's code
            # Some cells might have just delimiters or meta info, let's keep it simple
            cells.append(nbformat.v4.new_code_cell(cell_text))

    nb.cells = cells
    
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == "__main__":
    py_to_ipynb('/home/leondgarse/workspace/msi5102_group_project/customer-segmentation-eda-k-means-dbscan.py',
                 '/home/leondgarse/workspace/msi5102_group_project/customer-segmentation-eda-k-means-dbscan.ipynb')
