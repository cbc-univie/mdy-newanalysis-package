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

#run from root of repo!
# python .github/workflows/html2md.py
notebooks = glob.glob("docs/notebooks/*.ipynb")
for notebook in notebooks:
    print(f"Processing notebook {notebook}")
    nb = nbformat.read(notebook, nbformat.NO_CONVERT)
    #skip if first cell is not markdown
    if not nb.cells[0].cell_type == "markdown":
        print("Skipping...")
        continue
    source = nb.cells[0].source
    #there is a md button
    if "[![Open in Colab]" in source:
        print("Already an md badge. Skipping...")
        continue
    #there is a html section to edit
    elif "a href=" in source:
        print("Converting html badge to markdown badge...")
        html = nb.cells[0].source
        #print(nb.cells[0].source)
        md = html2md(html)
        nb.cells[0].source = md
        #print(nb.cells[0].source)
        nbformat.write(nb, notebook)
