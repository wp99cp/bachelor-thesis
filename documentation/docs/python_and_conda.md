# Python and Conda

## Conda

Conda is a package manager for Python and R. It is used to install and manage packages and environments.
For this project, I'm working with Python 3.10 and Miniconda.

::: tip Cheat Sheet
A nice conda cheat sheet can be
found [here](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf).
Or in that [YoutTube video](https://www.youtube.com/watch?v=23aQdrS58e0).
:::

::: details Installation Process

**Installation Process**

1) Download Miniconda (follow the instructions [here](https://docs.conda.io/en/latest/miniconda.html#installing)).
2) Install Miniconda `bash Miniconda3-py310_23.1.0-1-Linux-x86_64.sh`
3) Create conda environment `conda create --name bachelor_thesis python=3.10`
4) Switch to conda environment `conda activate bachelor_thesis`
5) Then to speed up conda, run the following commands (
   see [Libmamba](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community)).

```bash
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

6) Add Conda environment to Jupyter Notebook.
   See [this article](https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084).

```bash
conda install -c anaconda ipykernel
python -m ipykernel install --user --name bachelor_thesis
```

:::

## Switch Conda Environment

To switch between environments, use the following command:

```bash
conda activate bachelor_thesis
```

`bachelor_thesis` is the name of the environment I'm working with.

## Remove all Conda Packages in an Environment

To remove all packages in an environment, use the following command:

```bash
conda install --revision 0
```

## Save Conda Environment

To save the current environment, use the following command:

```bash
conda env export > environment.yml
```

## Restore Conda Environment

To restore the environment, use the following command:

```bash
conda env create -f environment.yml
```

