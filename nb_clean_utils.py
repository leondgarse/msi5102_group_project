import re

def clean_styles(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into cells
    parts = content.split('# %%')
    new_parts = []

    for part in parts:
        if part.strip().startswith('[markdown]'):
            lines = part.split('\n')
            header = lines[0] # Usually ' [markdown]'
            body_lines = lines[1:]
            
            # 1. Extract raw text from markdown cell
            raw_lines = []
            for line in body_lines:
                # We expect the line to start with '# ' or '#'
                if line.startswith('# '):
                    raw_lines.append(line[2:])
                elif line.startswith('#'):
                    raw_lines.append(line[1:])
                else:
                    raw_lines.append(line)
            
            raw_text = "\n".join(raw_lines)
            
            # 2. Clean HTML headers
            for i in range(1, 7):
                pattern = re.compile(fr'<h{i}.*?>(.*?)</h{i}>', re.DOTALL | re.IGNORECASE)
                raw_text = pattern.sub(lambda m: '#' * i + ' ' + m.group(1).strip(), raw_text)
            
            # 3. Strip other HTML
            raw_text = re.sub(r'<div.*?>', '', raw_text, flags=re.IGNORECASE | re.DOTALL)
            raw_text = re.sub(r'</div>', '', raw_text, flags=re.IGNORECASE)
            
            # 4. Canonicalize Headers (remove redundant hashes if any)
            # Find lines like '## # Title' and turn them into '## Title'
            raw_text = re.sub(r'(#+)\s*#+\s*', r'\1 ', raw_text)
            # Find lines like '### Title' where it should have been '# Title'
            # (Based on current messy state: line 3 was '# # # Analysis')
            # Let's just look for any sequence of hashes and text
            
            # 5. Clean consecutive empty lines and trailing ones
            final_lines = []
            prev_was_empty = False
            for line in raw_text.split('\n'):
                l = line.strip()
                if l:
                    # Strip ANY leading hashes from the content recursively to start fresh
                    stripped = l.lstrip('#').strip()
                    # Determine level if it was a header
                    m = re.match(r'^(#+)', l)
                    if m:
                        level = len(m.group(1))
                        if level > 6: level = 6
                        final_lines.append(f"# {'#' * level} {stripped}")
                    else:
                        final_lines.append(f"# {stripped}")
                    prev_was_empty = False
                else:
                    if not prev_was_empty:
                        final_lines.append("#")
                        prev_was_empty = True
            
            # Remove trailing empty lines
            while final_lines and final_lines[-1] == "#":
                final_lines.pop()
            # Remove leading empty lines
            while final_lines and final_lines[0] == "#":
                final_lines.pop(0)

            new_parts.append(header + "\n" + "\n".join(final_lines) + "\n")
        else:
            new_parts.append(part)

    final_content = "# %%".join(new_parts)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_content)

if __name__ == "__main__":
    clean_styles('/home/leondgarse/workspace/msi5102_group_project/customer-segmentation-eda-k-means-dbscan.py', 
                 '/home/leondgarse/workspace/msi5102_group_project/customer-segmentation-eda-k-means-dbscan.py')
    print("Cleaned styles and standardized headers.")
