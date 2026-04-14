import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath('../py'))

project = 'minimint'
author = 'Sergey Koposov'
copyright = f"{datetime.now().year}, {author}"

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
]

autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': False,
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
