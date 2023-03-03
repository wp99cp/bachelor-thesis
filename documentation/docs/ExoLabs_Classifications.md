# ExoLabs Classification

the Landsat data and some auxiliary data (an additional 10 m DEM will be added), which you can find
here: [Google Drive Folder](https://drive.google.com/drive/folders/1wUvIOBuAUKaGnc1AcOaviNbmsC5KEufV?usp=sharing).

## The classification looks as follows:

```
notObserved = 0      (-)     - no data
noData = 1          (grey)   - unknown
darkFeatures = 2    (black)  - unknown
clouds = 3          (white)  - unknown
snow = 4            (red)    - snow
vegetation = 5      (green)  - no snow
water = 6           (blue)   - no snow
bareSoils = 7       (yellow) - no snow
glacierIce = 8      (cyan)   - no snow
```

## The snow classification is based on the classification (just remapped) and looks as follows:

```
0 = ‘no data’
1 = ‘no snow’
2 = ‘snow’
3 = ‘unknown’
```