# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import datetime
from importlib import import_module
import sys
import tomllib
from pathlib import Path


extensions = ['sphinx_automodapi.automodapi', 
              'sphinx.ext.intersphinx'
              ]

intersphinx_mapping = {'python': ('https://docs.python.org/3', None)}

try:
    from sphinx_astropy.conf.v1 import *  # noqa
    print("Imported sphinx!")
except ImportError:
    print('ERROR: the documentation requires the sphinx-astropy package to be installed')
    sys.exit(1)

numpydoc_show_class_members = False

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Get configuration information from pyproject.toml
with (Path(__file__).parents[1] / "pyproject.toml").open("rb") as f:
    pyproject = tomllib.load(f)

project = pyproject["project"]["name"]
author = ",".join([l["name"] for l in pyproject["project"]["authors"]])
copyright = "{0}, {1}".format(datetime.datetime.now().year, author)

import_module(pyproject["project"]["name"])
package = sys.modules[pyproject["project"]["name"]]

# The short X.Y version.
version = package.__version__.split('-', 1)[0]
# The full version, including alpha/beta/rc tags.
release = package.__version__


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pyramid'
html_static_path = ["_static"]
html_sidebars = {
    '**': [
        'globaltoc.html',
    ]
}
