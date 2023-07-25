# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pykoi"
copyright = "2023, CambioML"
author = "CambioML"
release = "0.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
]

html_logo = "logo.svg"

html_theme_options = {
    "light_css_variables": {
        "color-admonition-title--note": "#00ebc7",
        "color-admonition-title-background--note": "#a2fff1",
    },
    "sidebar_hide_name": True,
    "footer_icons": [
        {
            "name": "Website",
            "url": "https://github.com/CambioML/pykoi",
            "html": """
                <svg width="447" height="425" viewBox="0 0 447 425" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M436 212.5C436 323.269 341.39 414 223.5 414C105.61 414 11 323.269 11 212.5C11 101.731 105.61 11 223.5 11C341.39 11 436 101.731 436 212.5Z" stroke="#393939" stroke-width="22"/>
                    <rect x="118" y="148" width="40" height="42.2633" rx="20" fill="#393939"/>
                    <rect x="210" y="148" width="40" height="42.2633" rx="20" fill="#393939"/>
                    <rect x="302" y="148" width="40" height="40" rx="20" fill="#393939"/>
                    <path d="M184.646 212C184.646 212 171.914 285 259.488 285C347.062 285 357 212 357 212" stroke="#393939" stroke-width="22" stroke-linecap="round"/>
                    <path d="M97.6462 212C97.6462 212 84.9137 285 172.488 285C260.062 285 270 212 270 212" stroke="#393939" stroke-width="22" stroke-linecap="round"/>
                </svg>
            """,
            "class": "",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/CambioML/pykoi",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ]
}
