# Build the docs
```bash
conda env create -f ../devtools/dev.yml
conda activate newanalysis-dev
make html
firefox _build/html/index.html
```

# Contributing to the API Docs
Use python docstrings directly in the pyx files to document the different functions.

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

**Note:**

The uploaded jupyter notebooks will automatically recieve an 'Open in Colab' button.
You do not have to do this manually, but in order to work in colab, please add the following to a new notebook prior to importing MDAnalysis and newanalysis:
```python
IN_COLAB = 'google.colab' in sys.modules
HAS_NEWANALYSIS = 'newanalysis' in sys.modules
if IN_COLAB:
    if not'MDAnalysis' in sys.modules:
        !pip install MDAnalysis
    import os
    if not os.path.exists("/content/mdy-newanalysis-package/"):
        !git clone https://github.com/cbc-univie/mdy-newanalysis-package.git
    !pwd
    if not HAS_NEWANALYSIS:
        %cd /content/mdy-newanalysis-package/newanalysis_source/
        !pip install .
    %cd /content/mdy-newanalysis-package/docs/notebooks/
```

