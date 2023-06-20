# ExoLabs Ground Classification

ExoLabs is a Spin-Off of the University of Zurich offering commercial snow coverage maps.

They use a decision tree to evaluate the spectral properties of different land cover types to assign probabilities to
eight different classes of ground coverage. This makes it similar in functionality to Fmask, but designed specifically
for snow analysis. It's a mono-temporal algorithm.

## Ground Classification

ExoLabs provides a ground classification for each image. The classification is based on the following classes:

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

## Snow Classification

The snow classification is based on the classification (just remapped) and looks as follows:

```
0 = ‘no data’
1 = ‘no snow’
2 = ‘snow’
3 = ‘unknown’
```