# Sentinel2 Bands

There are 13 Sentinel 2 bands in total. Each band is 10, 20, or 60 meters in pixel size.
Sentinel 2 consists of 2 satellites. First came Sentinel 2A which was launched in 2015. Next came Sentinel 2b in 2017.
Two additional satellites (Sentinel 2C and 2D) are planned to launch in 2024. This will make a total of four Sentinel-2
satellites.


::: info

Sentinel-2 is an Earth observation mission from the Copernicus Programme that systematically acquires optical imagery at
high spatial resolution over land and coastal waters. See [Wikipedia](https://en.wikipedia.org/wiki/Sentinel-2).

:::

## Overview of all Bands and Some Combining Examples

Sentinel-2 carries the Multispectral Imager (MSI). This sensor delivers 13 spectral bands ranging from 10 to 60-meter
pixel size.

- Its blue (B2), green (B3), red (B4), and near-infrared (B8) channels have a 10-meter resolution.
- Next, its red edge (B5), near-infrared NIR (B6, B7, and B8A), and short-wave infrared SWIR (B11 and B12) have a ground
  sampling distance of 20 meters.
- Finally, its coastal aerosol (B1) and cirrus band (B10) have a 60-meter pixel size.

::: details Sources

- [Sentinel: Spatial Resolution](https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial)
- [How many Spectral Bands have the Sentinel 2 Images?](https://hatarilabs.com/ih-en/how-many-spectral-bands-have-the-sentinel-2-images)
- [Sentinel 2 Bands and Combinations](https://gisgeography.com/sentinel-2-bands-combinations/)

:::

| Sentinel-2 Bands | Description                                                | Central Wavelength (µm) | Resolution (m) |
|------------------|------------------------------------------------------------|-------------------------|----------------|
| Band 1           | Ultra Blue (Coastal and Aerosol), Aerosol detection        | 0.443                   | 60             |
| Band 2           | Blue                                                       | 0.490                   | 10             |
| Band 3           | Green                                                      | 0.560                   | 10             |
| Band 4           | Red                                                        | 0.665                   | 10             |
| Band 5           | Visible and Near Infrared (VNIR), Vegetation Red Edge      | 0.705                   | 20             |
| Band 6           | Visible and Near Infrared (VNIR), Vegetation Red Edge      | 0.740                   | 20             |
| Band 7           | Visible and Near Infrared (VNIR), Vegetation Red Edge      | 0.783                   | 20             |
| Band 8           | Visible and Near Infrared (VNIR)                           | 0.842                   | 10             |
| Band 8A          | Visible and Near Infrared (VNIR), Vegetation Red Edge      | 0.865                   | 20             |
| Band 9           | Short Wave Infrared (SWIR) - Detecting Water vapour        | 0.945                   | 60             |
| Band 10          | Short Wave Infrared (SWIR) - Cirrus Cloud detection        | 1.375                   | 60             |
| Band 11          | Short Wave Infrared (SWIR) - Snow/Ice/Cloud discrimination | 1.610                   | 20             |
| Band 12          | Short Wave Infrared (SWIR) - Snow/Ice/Cloud discrimination | 2.190                   | 20             |

## Combinations

We use band combinations to better understand the features in imagery. The way we do this is by rearranging the
available channels in creative ways.

| Band Combination      | Name                | Description                                                                                                                                                                                                                                                                                                                                                  |
|-----------------------|---------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| B2/B3/B4              | Natural Colo        | RGB (Red, Green, Blue),  its purpose is to display imagery the same way our eyes see the world. Next, urban features often appear white and grey. Finally, water is a shade of dark blue depending on how clean it is.                                                                                                                                       |
| B3/B4/B8              | Color Infrared      | False Color Infrared (FCI), this combination is used to highlight healthy and unhealthy vegetation. By using the near-infrared (B8) band, it’s especially good at reflecting chlorophyll. This is why in a color infrared image, denser vegetation is red. But urban areas are white.                                                                        |
| B12/B8A/B4            | Short-Wave Infrared | The short-wave infrared band combination uses SWIR (B12), NIR (B8A), and red (B4). This composite shows vegetation in various shades of green. In general, darker shades of green indicate denser vegetation. But brown is indicative of bare soil and built-up areas.                                                                                       |
| B11/B8/B2             | Agriculture         | The agriculture band combination uses SWIR-1 (B11), near-infrared (B8), and blue (B2). It’s mostly used to monitor the health of crops because of how it uses short-wave and near-infrared. Both these bands are particularly good at highlighting dense vegetation that appears as dark green.                                                              |
| B12/B11/B2            | Geology             | The geology band combination is a neat application for finding geological features. This includes faults, lithology, and geological formations. By leveraging the SWIR-2 (B12), SWIR-1 (B11), and blue (B2) bands, geologists tend to use this Sentinel band combination for their analysis.                                                                 |
| B4/B3/B1              | Bathymetric         | As the name implies, the bathymetric band combination is good for coastal studies. The bathymetric band combination uses the red (B4), green (B3), and coastal band (B1). Using the coastal aerosol band is good for estimating suspended sediment in the water.                                                                                             |
| (B8-B4) / (B8+B4)     | Vegetation Index    | Because near-infrared (which vegetation strongly reflects) and red light (which vegetation absorbs), the vegetation index is good for quantifying the amount of vegetation. The formula for the normalized difference vegetation index is (B8-B4)/(B8+B4). While high values suggest dense canopy, low or negative values indicate urban and water features. |
| (B8A-B11) / (B8A+B11) | Moisture Index      | The moisture index is ideal for finding water stress in plants. It uses the short-wave and near-infrared to generate an index of moisture content. In general, wetter vegetation has higher values. But lower moisture index values suggest plants are under stress from insufficient moisture.                                                              |
| B11                   | Clouds              | band B11 with Contrast set to 50 and Gamma to 2                                                                                                                                                                                                                                                                                                              |

See
also [Crop Classification using Multi-spectral and Multitemporal Satellite Imagery with Machine Learning; Crop Classification using Multi-spectral and Multitemporal Satellite Imagery with Machine Learning](https://ieeexplore.ieee.org/document/8903738)
for more combinations.