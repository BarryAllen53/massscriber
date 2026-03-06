# Releasing Massscriber

Massscriber uses Semantic Versioning.

## Version source of truth

The package version lives in:

```python
massscriber.__version__
```

`pyproject.toml` reads that value dynamically during packaging.

## Release checklist

1. Update `massscriber/__init__.py` with the new version.
2. Move the relevant notes from `Unreleased` into a dated section in `CHANGELOG.md`.
3. Commit the release preparation changes.
4. Create an annotated Git tag:

```powershell
git tag -a v0.1.0 -m "Release v0.1.0"
```

5. Push the branch and tags:

```powershell
git push origin main
git push origin --tags
```

## What happens after pushing a tag

The GitHub Actions release workflow will:

- install the project
- run tests
- build a wheel and source distribution
- verify that the tag version matches `massscriber.__version__`
- create a GitHub Release
- upload the built artifacts to that release
- publish the tagged version to PyPI through Trusted Publishing once the publisher is configured on PyPI

## PyPI setup

Massscriber uses PyPI Trusted Publishing via GitHub Actions OIDC.

One-time setup on PyPI:

1. Open the project's publishing settings while logged into PyPI as a project owner.
2. Add a GitHub publisher for:
   - owner: `BarryAllen53`
   - repository: `massscriber`
   - workflow: `pypi-publish.yml`
   - environment: `pypi`

After that, the GitHub workflow can publish without storing a long-lived PyPI token in GitHub secrets.

## Tag format

Release tags must use this format:

```text
vMAJOR.MINOR.PATCH
```

Example:

```text
v0.1.0
```
