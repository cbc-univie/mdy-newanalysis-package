import glob
import nbformat

def html2md(html):
    url, img = None, None
    for entry in html.split("><"):
        if entry.startswith("a href="):
            url = entry.split('"')
        elif entry.startswith("img src="):
            img = entry.split('"')[1]
    if url is None or img is None:
        print(url)
        print(img)
        raise RuntimeError("There wa a problem in the conversion")
    md = f"[![Open in Colab]({img})]({url})" 
    return md

notebooks = glob.glob("../../docs/notebooks/*.ipynb")
for notebook in notebooks:
    nb = nbformat.read(notebook, nbformat.NO_CONVERT)
    #skip if first cell is not markdown
    if not nb.cells[0].cell_type == "markdown":
        continue
    source = nb.cells[0].source
    #there is a md button
    if source.contains("[![Open in Colab]"):
        continue
    #there is a html section to edit
    elif source.contains("a href="):
        html = nb.cells[0].source
        print(nb.cells[0].source)
        md = html2md(html)
        nb.cells[0].source = md
        print(nb.cells[0].source)
        nbformat.write(nb, notebook)
