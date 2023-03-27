# Build the docs
```bash
conda env create -f ../devtools/dev.yml
conda activate newanalysis-dev
make html
firefox _build/html/index.html
```

# Contributing to the gallery
Create a new jupyter notebook in `notebooks`.

```bash
jupyter-notebook 
```

Probably you want to make your newanalysis-dev kernel available for jupyter:
```bash
conda activate newanalysis-dev
ipython kernel install --name "newanalysis-dev" --user
```
Then you should be able to select it in the webinterface of jupyter.

**Add new notebook to the documentation**

Add the following line to gallery.rst:
(below the already existing ones)

```notebooks/filename_withoutextension```

