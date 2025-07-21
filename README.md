[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/snapenv/snap-package-template) [![Open in GitHub Codespaces](https://img.shields.io/static/v1?label=GitHub%20Codespaces&message=Open&color=blue&logo=github)](https://github.com/snapenv/snap-package-template)

# Snap Package Template

SnapEnv template for Python package.

# About this template

This template repository provides the boilerplate to create a python package.

It was created having in mind SnapEnv people and what the most common use-cases would be. Following its structure you'll get into developing your next FastAPI server in no time!

## 🎁 Features

- 🧑‍💻 Quick and reproducible development environments with VS Code's [Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers), PyCharm's [Docker Compose interpreter](https://www.jetbrains.com/help/pycharm/using-docker-compose-as-a-remote-interpreter.html#docker-compose-remote), and [GitHub Codespaces](https://github.com/features/codespaces)
- 🌈 Cross-platform support for Linux, macOS (Apple silicon and Intel), and Windows
- 🐚 Modern shell prompt with [Starship](https://github.com/starship/starship)
- 📦 Packaging and dependency management with [Poetry](https://github.com/python-poetry/poetry)
- 📃 Documentation structure using [mkdocs](http://www.mkdocs.org)
- ⚡️ Task running with [Poe the Poet](https://github.com/nat-n/poethepoet)
- ✍️ Code formatting with [Ruff](https://github.com/charliermarsh/ruff)
- ✅ Code linting with [Pre-commit](https://pre-commit.com/), [Mypy](https://github.com/python/mypy), and [Ruff](https://github.com/charliermarsh/ruff)
- 🏷 Optionally follows the [Conventional Commits](https://www.conventionalcommits.org/) standard to automate [Semantic Versioning](https://semver.org/) and [Keep A Changelog](https://keepachangelog.com/) with [Commitizen](https://github.com/commitizen-tools/commitizen)
- 💌 Verified commits with [GPG](https://gnupg.org/)
- ♻️ Continuous integration with [GitHub Actions](https://docs.github.com/en/actions) or [GitLab CI/CD](https://docs.gitlab.com/ee/ci/)
- 🧪 Test coverage with [Coverage.py](https://github.com/nedbat/coveragepy)

## Getting Started

> **DO NOT CLONE OR FORK**

If you want to set up this template:

1. Request a repository on SnapEnv GitHub by following the standard procedure on the wiki. It will install the template directly. Alternatively, set it up in your personal GitHub account by clicking **[Use this template](https://github.com/snapenv/snap-package-template/generate)**.
2. Wait until the first run of CI finishes. Github Actions will commit to your new repo with a "refactor: 🎉 Ready to clone and code." message.
3. Uset git to clone your new repository.
4. Delete optional files: 
    - If you don't need automatic documentation generation, you can delete folder `docs`, `mkdocs-overrides`, file `.github\workflows\github-pages-dev.yml`, `.github\workflows\reusable-github-pages.yml` and `mkdocs.yml`
    - If you don't want automatic testing, you can delete folder `tests` and file `.github\workflows\test.yml`
5. Prepare a virtual environment and test:
```bash
poetry install
poetry shell
ENVIRONMENT=dev poe api --dev
```
6. Adapt anything else to your project. 
7. Read the file [docs/index.md](docs/index.md) for more information about development of your server.

## Contributing

We love all types of contributions: whether big or small helping in improving this community resource.

A contribution can be as small as a ⭐ or even finding and creating issues.

1. There are a number of [open issues present](https://github.com/snapenv/snap-package-template/issues) which can be good ones to start with
2. If you have suggestions for enhancements, wish to contribute a simple fix such as correcting a typo, or want to address an apparent bug, please feel free to initiate a new issue or submit a pull request. See [CONTRIBUTING.md](CONTRIBUTING.md) to learn how to setup a local development environment for this repository.
2. If you're contemplating a larger change or addition to this repository, be it in terms of its structure or the features, kindly begin by creating a new issue [open a new issue :octocat:](https://github.com/snapenv/snap-package-template/issues/new) and outline your proposed changes. This will allow us to engage in a discussion before you dedicate a significant amount of time or effort. Your cooperation and understanding are appreciated

## Credits

I adapted this template from these excellent sources:

- [poetry-cookiecutter](https://github.com/radix-ai/poetry-cookiecutter) by [radix-ai](https://github.com/radix-ai)
- [ukp-project-template](https://github.com/UKPLab/ukp-project-template) by [UKPLab](https://github.com/UKPLab)
- [python-project-template](https://github.com/rochacbruno/python-project-template/) by [rochacbruno](https://github.com/rochacbruno)
- [bootstrap-python-package](https://github.com/febus982/bootstrap-python-package) by [febus982](https://github.com/febus982)

