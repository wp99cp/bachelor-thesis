# This Page Describes the Process of Hand Annotations Image

The existing data sets are not sufficient to train (especially to verify) our model in a reliable way. Therefore, we
need to create a new data set with hand annotations. This process is tedious and time consuming, but it is
necessary to create a reliable model.

## Annotating Process

I've created a web application helps to annotate the images. The workflow is as follows:

1) Upload a scene (including all sentinel bands, ExoLab classification and additional meta-data) to the server.

2) Work throw the scene, the backend of the web application will save the annotations and create two `.jp2`
   files. `mask_coverage.jp2` and `mask_snow.jp2`. The first one is a binary mask, where `0` means no data mask was
   created for the given pixel and `1` means a mask was created. The second one contains the annotations.

3) Once a scene is sufficiently annotated, we can download the annotations. The most convenient way to do this is by
   accessing the following URL: `https://backend.annotator.emeal.ch//download/masks`. The backend will create a
   `.zip` file with all the annotations.

4) In the next step, open the results in QGIS and do a visual inspection. For that create a New Shapefile layer, within
   that layers we mark points of areas that need to be revised. Save the layer as a `GeoJSON` file. The file should
   be saved using the `EPSG:326332 - WGS 84 / UTM zone 32N` projection.

5) Add the file (which should be called `revisions.geojson`) to the corresponding folder. The application will
   automatically detect the file and will switch to the revision mode (you may need to restart the application).

## Classes in our Annotation

```
0 - background, no special class
1 - snow
2 - dense clouds
3 - water
4 - semi-transparent clouds
```

## Annotation Software

The annotation application is located under `/pre-processing/image_annotator`. It is a web application written in
`python` and `flask`. The frontend is written using `angular` and `typescript`.

To deploy the application, you need to install `docker` and `docker-compose`.

```bash
docker-compose build && docker-compose push
```

### Execution mode

You can run the application in two different modes:

- Annotation of new areas
- Annotation of areas that need to be revised

The first mode is the default mode. To run the application in the second mode, you need to add a GeoJSON
file to the corresponding mask folder. The file should be called `revisions.geojson` and should be located
