# Working with the documentation

[mkdocs](https://www.mkdocs.org/) and [mkdocs-material](https://squidfunk.github.io/mkdocs-material) are used to create the documentation of this project.

## Local

1. Install dev dependencies (this install mkdocs and dependencies)
`uv pip install -e .[test]`
2. Create the documentation and start webserver serving them
```
mkdocs server -a localhost:5555
```
or create the html pages with
```
mkdocs build -d docs_build
```

## Upload documentation to github-pages

[Guide](https://www.mkdocs.org/user-guide/deploying-your-docs/)
