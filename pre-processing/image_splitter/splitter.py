import datetime
import json
import subprocess

import cv2


def extract_metadata(filename):
    # the capture date is in the file name
    capture_date = filename.split('_')[1]
    date = datetime.datetime.strptime(capture_date, "%Y%m%dT%H%M%S")

    bash_command = "gdalinfo " + filename + " -json"
    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    gdal_info_json = json.loads(output.decode('utf-8'))

    metadata = {
        'capture_date': date,
        'gdal_info': gdal_info_json
    }

    return metadata


def main():
    filepath = 'input/T32TMS_20221018T102031_TCI.jp2'
    print('Reading image from: ', filepath)

    metadata = extract_metadata(filepath)
    print(metadata)

    # Convert
    image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

    # split the image into images of size 512x512
    for i in range(0, image.shape[0], 512):
        for j in range(0, image.shape[1], 512):

            # continue if the image is not 512x512
            if i + 512 > image.shape[0] or j + 512 > image.shape[1]:
                continue

            date = metadata['capture_date'].date()
            part_i = i // 512
            part_j = j // 512
            cv2.imwrite('output/{}_{}_{}.png'.format(part_i, part_j, date), image[i:i + 512, j:j + 512])


if __name__ == '__main__':
    main()
