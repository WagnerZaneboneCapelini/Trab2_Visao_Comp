[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/snapenv/snap-package-template) [![Open in GitHub Codespaces](https://img.shields.io/static/v1?label=GitHub%20Codespaces&message=Open&color=blue&logo=github)](https://github.com/snapenv/snap-package-template)

![Static Badge](https://img.shields.io/badge/Python-33.11_%7C_3.12-blue?logo=python&logoColor=white)
[![Stable Version](https://img.shields.io/pypi/v/bootstrap-python-package?color=blue)](https://pypi.org/project/bootstrap-python-package/)
[![stability-beta](https://img.shields.io/badge/stability-beta-33bbff.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#beta)

[![Python tests](https://github.com/febus982/bootstrap-python-package/actions/workflows/python-tests.yml/badge.svg?branch=main)](https://github.com/febus982/bootstrap-python-package/actions/workflows/python-tests.yml)
[![Maintainability](https://api.codeclimate.com/v1/badges/593e78ec96ed5ebb0dd3/maintainability)](https://codeclimate.com/github/febus982/bootstrap-python-package/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/593e78ec96ed5ebb0dd3/test_coverage)](https://codeclimate.com/github/febus982/bootstrap-python-package/test_coverage)

[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)

# 👖 SnapEnv template for Python package

SnapEnv template for Python package.

It was created having in mind SnapEnv people and what the most common use-cases would be. Following its structure you'll get into developing your next FastAPI server in no time!

- a configuration module made using pydantic-settings
- a handful of Pydantic schemas to make your life as a programmer easier

## Features

1. 🍬 *Sklearn meta-estimator*: add conformal prediction of quantiles and intervals to any scikit-learn regressor
2. 🔮 *Darts forecaster:* add conformally calibrated probabilistic forecasting to any scikit-learn regressor
3. 🌡️ *Conformally calibrated:* accurate quantiles, and intervals with reliable [coverage](https://en.wikipedia.org/wiki/Coverage_probability)
4. 🚦 *Coherent quantiles:* quantiles increase monotonically instead of [crossing](https://github.com/dmlc/xgboost/issues/9848) [each other](https://github.com/microsoft/LightGBM/issues/3447)
5. 👖 *Tight quantiles:* selects the lowest [dispersion](https://en.wikipedia.org/wiki/Statistical_dispersion) that provides the desired coverage
6. 🎁 *Data efficient:* requires only a small number of calibration examples to fit
7. 🐼 *Pandas support:* optionally predict on DataFrames and receive DataFrame output

## Using

### Quick links

1. [Installing](#installing)

### Installing

```sh
poetry add snap-package-template
```

```sh
pip install snap-package-template
```

## Contributing

/// details | Prerequisites
1. Set up Git to use SSH
    1. [Generate an SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key) and [add the SSH key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

    2. Configure SSH to automatically load your SSH keys:
        ```sh
        cat << EOF >> ~/.ssh/config
        
        Host *
          AddKeysToAgent yes
          IgnoreUnknown UseKeychain
          UseKeychain yes
          ForwardAgent yes
        EOF
        ```

2. [Install Docker Desktop](https://www.docker.com/get-started).
    - _Linux only_:
        - Export your user's user id and group id so that [files created in the Dev Container are owned by your user](https://github.com/moby/moby/issues/3206):
            ```sh
            cat << EOF >> ~/.bashrc
            
            export UID=$(id --user)
            export GID=$(id --group)
            EOF
            ```

3. Install VS Code or PyCharm
    1. [Install VS Code](https://code.visualstudio.com/) and [VS Code's Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers). Alternatively, install [PyCharm](https://www.jetbrains.com/pycharm/download/).
    2. _Optional:_ install a [Nerd Font](https://www.nerdfonts.com/font-downloads) such as [FiraCode Nerd Font](https://github.com/ryanoasis/nerd-fonts/tree/master/patched-fonts/FiraCode) and [configure VS Code](https://github.com/tonsky/FiraCode/wiki/VS-Code-Instructions) or [configure PyCharm](https://github.com/tonsky/FiraCode/wiki/Intellij-products-instructions) to use it.

///

/// details | Development environments

The following development environments are supported:

1. ⭐️ _GitHub Codespaces_: click on _Code_ and select _Create codespace_ to start a Dev Container with [GitHub Codespaces](https://github.com/features/codespaces).
1. ⭐️ _Dev Container (with container volume)_: click on [Open in Dev Containers](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/snapenv/snap-backend-template) to clone this repository in a container volume and create a Dev Container with VS Code.
1. _Dev Container_: clone this repository, open it with VS Code, and run <kbd>Ctrl/⌘</kbd> + <kbd>⇧</kbd> + <kbd>P</kbd> → _Dev Containers: Reopen in Container_.
1. _PyCharm_: clone this repository, open it with PyCharm, and [configure Docker Compose as a remote interpreter](https://www.jetbrains.com/help/pycharm/using-docker-compose-as-a-remote-interpreter.html#docker-compose-remote) with the `dev` service.
1. _Terminal_: clone this repository, open it with your terminal, and run `docker compose up --detach dev` to start a Dev Container in the background, and then run `docker compose exec dev zsh` to open a shell prompt in the Dev Container.

///

/// details | Developing

- This project follows the [Conventional Commits](https://www.conventionalcommits.org/) standard to automate [Semantic Versioning](https://semver.org/) and [Keep A Changelog](https://keepachangelog.com/) with [Commitizen](https://github.com/commitizen-tools/commitizen).
- Run `poe` from within the development environment to print a list of [Poe the Poet](https://github.com/nat-n/poethepoet) tasks available to run on this project.
- Run `poetry add {package}` from within the development environment to install a run time dependency and add it to `pyproject.toml` and `poetry.lock`. Add `--group test` or `--group dev` to install a CI or development dependency, respectively.
- Run `poetry update` from within the development environment to upgrade all dependencies to the latest versions allowed by `pyproject.toml`.
- Run `poe docs`, `poe lint` and `ENVIRONMENT=dev poe test` before any commit, or your git push can fail. `poe docs` generate any new documentation for changes/additions in the python modules.
- Run `cz --name cz_gitmoji commit` to commit files using conventional commits with emojis.
- Run `cz --name cz_gitmoji bump --changelog` to bump the package's version, update the `CHANGELOG.md`, and create a git tag.
- Run `git push --tags` to push the new tag to github.

///
