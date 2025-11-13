# Working with the documentation

[mkdocs](https://www.mkdocs.org/) and [mkdocs-material](https://squidfunk.github.io/mkdocs-material) are used to create the documentation of this project.

The configuration of mkdocs is done with `mkdocs.yml` in the root of this repository. There you can add files to the documentation, manage the plugins and the navigation pane.

## Local

1. Install dev dependencies (this install mkdocs and dependencies)
`uv pip install -e .[test]`
2. Create the documentation and start webserver serving them
```
mkdocs serve -a localhost:5555
```
or create the html pages with
```
mkdocs build -d docs_build
```

## Upload documentation to github-pages
You do not need to manually update the github pages. The update is performed by the docs github workflow defined in `.github/workflows/docs.yml`.