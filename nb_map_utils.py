
import nbformat
import sys

def analyze_full_notebook(path):
    print(f"Mapping: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    print(f"Total cells: {len(nb.cells)}")
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown':
            content = cell.source.strip()
            if content.startswith('#') or content.startswith('<h1') or content.startswith('<h2') or content.startswith('<h3'):
                print(f"Index {i}: [MD] {content.split('\\n')[0][:100]}")
        elif cell.cell_type == 'code':
             first_line = cell.source.split('\n')[0].strip()
             if first_line:
                 print(f"Index {i}: [Code] {first_line[:100]}")

if __name__ == "__main__":
    path = '/home/leondgarse/workspace/msi5102_group_project/customer-segmentation-eda-k-means-dbscan.ipynb'
    if len(sys.argv) > 1:
        path = sys.argv[1]
    analyze_full_notebook(path)
