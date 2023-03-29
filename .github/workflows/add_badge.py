import glob
import nbformat
from nbformat.v4 import new_markdown_cell

notebooks = glob.glob("../../docs/notebooks/*.ipynb")
for notebook in notebooks:
    print(f"Processing notebook {notebook}")
    nb = nbformat.read(notebook, nbformat.NO_CONVERT)
    #skip if first cell is not markdown
    if not nb.cells[0].cell_type == "markdown":
        print("Skipping...")
        continue
    source = nb.cells[0].source
    #there is already a badge or button
    if source.contains("{{ badge }}") or source.contains("Open in Colab"):
        print("Skipping...")
        continue
    print("Inserting {{ badge }}")
    nb.cells.insert(0, new_markdown_cell(source='{{ badge }}'))
    nbformat.write(nb, notebook)
