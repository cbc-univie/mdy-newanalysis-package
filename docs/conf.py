# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import mock

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'newanalysis'
copyright = '2023, cbc-univie'
author = 'cbc-univie'

version = '0.1'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx_gallery.load_style",
]

napoleon_google_docstring = True

templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'

sys.path.insert(0, os.path.abspath('../newanalysis'))
MOCK_MODULES = ["newanalysis/"]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Thumbnail selection for nbsphinx gallery
nbsphinx_thumbnails = {
    #default: would overwrite notebook automatic generated thumbnails
    #"notebooks/*": "_static/thumbnails/default.png",
    #specify your specific notebook and thumnail here:
    "notebooks/rdf": "_static/thumbnails/default.png",
    #you may also specify a specific cell using the metadata in jupyter:
    # view -> cell toolbar -> tags: put nbsphinx-thumbnail there
    # https://nbsphinx.readthedocs.io/en/0.9.1/gallery/cell-tag.html
}

# -- Options for todo extension ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#configuration

todo_include_todos = True
