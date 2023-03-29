import glob
import nbformat
from nbformat.v4 import new_markdown_cell
import logging
logging.setLevel(20) #Info

notebooks = glob.glob("../../docs/notebooks/*.ipynb")
for notebook in notebooks:
    logging.info(f"Processing notebook {notebook}")
    nb = nbformat.read(notebook, nbformat.NO_CONVERT)
    #skip if first cell is not markdown
    if not nb.cells[0].cell_type == "markdown":
        logging.info("Skipping...")
        continue
    source = nb.cells[0].source
    #there is already a badge or button
    if "{{ badge }}" in source or "open in colab" in source.lower():
        logging.info("Skipping...")
        continue
    logging.info("Inserting {{ badge }}")
    nb.cells.insert(0, new_markdown_cell(source='{{ badge }}'))
    nbformat.write(nb, notebook)
