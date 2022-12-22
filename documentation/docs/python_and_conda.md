# Python and Conda

## Conda

Conda is a package manager for Python and R. It is used to install and manage packages and environments.
For this project, I'm working with Python 3.10.8 and Conda 22.9.0.

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

