# Data Sources and Overview

The project utilizes multiple data sources both for satellite images and for automatically annotated masks.

- [Data Pre-Processing](/docs/working_pipeline/pipeline)

The following table gives an overview of the data sources and their usage.

| Data Source / Data Set | Usage in this Project | Link with Details                         |
|------------------------|-----------------------|-------------------------------------------|
| Sentinel-2             | Satellite Images      | [Sentinel-2](/docs/datasources/sentinel2) |
| Landsat-8              | Satellite Images      | [Landsat-8](/docs/datasources/landsat8)   |
| ExoLabs                | Masks                 | [ExoLabs](/docs/datasources/exolabs)      |
| Water Models           | Masks Creation        | -                                         |
| Elevation Models       | Auxiliary Data        | -                                         |

::: info

The data can be found either in
the [Google Drive Folder](https://drive.google.com/drive/folders/1wUvIOBuAUKaGnc1AcOaviNbmsC5KEufV?usp=sharing) or it is
stored in `pf/pfstud/nimbus/downloaded_data`.

:::

## Additional Dataset (Not Used in this Project)

The following table gives an overview of additional data sources that are not used in this project.

![summary_of_datasets_available.png](../images/summary_of_datasets_available.png)
![quality_of_available_datasets.png](../images/quality_of_available_datasets.png)

Source: Skakun et al. (2022), Cloud Mask Intercomparison eXercise (CMIX): An evaluation of cloud masking algorithms for
Landsat 8 and Sentinel-2
