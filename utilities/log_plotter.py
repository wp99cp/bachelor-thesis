import argparse

from matplotlib import pyplot as plt


def main():
    """
    Given a log file path (as a command line argument), plot the loss and metrics values.
    """

    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=str, help="Path to the log file.")
    args = parser.parse_args()

    # read the log file
    file = open(args.log_file, "r")
    lines = file.readlines()
    file.close()

    # extract the loss and metrics values
    train_loss = []
    test_loss = []

    pixel_accuracy_background = []
    pixel_accuracy_snow = []
    pixel_accuracy_clouds = []
    pixel_accuracy_water = []

    union_over_inter_background = []
    union_over_inter_snow = []
    union_over_inter_clouds = []
    union_over_inter_water = []

    dice_background = []
    dice_snow = []
    dice_clouds = []
    dice_water = []

    for i, line in enumerate(lines):

        # continue until the next epoch is reached (Epoch: 1, train_loss:)
        if not line.startswith("Epoch: "):
            continue

        # parse line Epoch: 1, train_loss: 0.294373, test_loss: 0.158697, EStop:
        train_loss += [float(line.split(", train_loss: ")[1].split(", test_loss: ")[0])]
        test_loss += [float(line.split(", test_loss: ")[1].split(", EStop:")[0])]

        # skip the next line
        line = lines[i + 2]
        pixel_accuracy_background += [float(line.split("pixel_accuracy___background: ")[1].split(",")[0])]

        line = lines[i + 3]
        union_over_inter_background += [float(line.split("union_over_inter_background: ")[1].split(",")[0])]

        line = lines[i + 4]
        dice_background += [float(line.split("dice_coefficient_background: ")[1].split(",")[0])]

        line = lines[i + 5]
        pixel_accuracy_snow += [float(line.split("pixel_accuracy___snow: ")[1].split(",")[0])]

        line = lines[i + 6]
        union_over_inter_snow += [float(line.split("union_over_inter_snow: ")[1].split(",")[0])]

        line = lines[i + 7]
        dice_snow += [float(line.split("dice_coefficient_snow: ")[1].split(",")[0])]

        line = lines[i + 8]
        pixel_accuracy_clouds += [float(line.split("pixel_accuracy___clouds: ")[1].split(",")[0])]

        line = lines[i + 9]
        union_over_inter_clouds += [float(line.split("union_over_inter_clouds: ")[1].split(",")[0])]

        line = lines[i + 10]
        dice_clouds += [float(line.split("dice_coefficient_clouds: ")[1].split(",")[0])]

        line = lines[i + 11]
        pixel_accuracy_water += [float(line.split("pixel_accuracy___water: ")[1].split(",")[0])]

        line = lines[i + 12]
        union_over_inter_water += [float(line.split("union_over_inter_water: ")[1].split(",")[0])]

        line = lines[i + 13]
        dice_water += [float(line.split("dice_coefficient_water: ")[1].split(",")[0])]

    # plot the loss and metrics values
    plt.figure(figsize=(20, 10))

    plt.subplot(2, 3, 1)
    plt.plot(train_loss, label="train")
    plt.plot(test_loss, label="test")
    plt.title("Loss")
    plt.legend()

    # mark the minimum loss with a thin dashed vertical line
    plt.axvline(train_loss.index(min(train_loss)), color="blue", linewidth=0.7, linestyle="--")
    plt.axvline(test_loss.index(min(test_loss)), color="orange", linewidth=0.7, linestyle="--")

    # set y axis to log scale
    plt.yscale("log")

    plt.subplot(2, 3, 2)
    plt.plot(pixel_accuracy_background, label="background")
    plt.plot(pixel_accuracy_snow, label="snow")
    plt.plot(pixel_accuracy_clouds, label="clouds")
    plt.plot(pixel_accuracy_water, label="water")
    plt.title("Pixel Accuracy")
    plt.legend()

    # mark each maximum level with a thin dashed horizontal line in the corresponding color
    plt.axhline(max(pixel_accuracy_background), color="blue", linewidth=0.7, linestyle="--")
    plt.axhline(max(pixel_accuracy_snow), color="orange", linewidth=0.7, linestyle="--")
    plt.axhline(max(pixel_accuracy_clouds), color="green", linewidth=0.7, linestyle="--")
    plt.axhline(max(pixel_accuracy_water), color="red", linewidth=0.7, linestyle="--")

    # mark each maximum level with a thin dashed vertical line in the corresponding color
    plt.axvline(pixel_accuracy_background.index(max(pixel_accuracy_background)), color="blue", linewidth=0.7, linestyle="--")
    plt.axvline(pixel_accuracy_snow.index(max(pixel_accuracy_snow)), color="orange", linewidth=0.7, linestyle="--")
    plt.axvline(pixel_accuracy_clouds.index(max(pixel_accuracy_clouds)), color="green", linewidth=0.7, linestyle="--")
    plt.axvline(pixel_accuracy_water.index(max(pixel_accuracy_water)), color="red", linewidth=0.7, linestyle="--")

    plt.subplot(2, 3, 3)
    plt.plot(union_over_inter_background, label="background")
    plt.plot(union_over_inter_snow, label="snow")
    plt.plot(union_over_inter_clouds, label="clouds")
    plt.plot(union_over_inter_water, label="water")
    plt.title("Union Over Intersection")
    plt.ylim(0.5, 1)
    plt.legend()

    # mark each maximum level with a thin dashed horizontal line in the corresponding color
    plt.axhline(max(union_over_inter_background), color="blue", linewidth=0.7, linestyle="--")
    plt.axhline(max(union_over_inter_snow), color="orange", linewidth=0.7, linestyle="--")
    plt.axhline(max(union_over_inter_clouds), color="green", linewidth=0.7, linestyle="--")
    plt.axhline(max(union_over_inter_water), color="red", linewidth=0.7, linestyle="--")

    # mark each maximum level with a thin dashed vertical line in the corresponding color
    plt.axvline(union_over_inter_background.index(max(union_over_inter_background)), color="blue", linewidth=0.7, linestyle="--")
    plt.axvline(union_over_inter_snow.index(max(union_over_inter_snow)), color="orange", linewidth=0.7, linestyle="--")
    plt.axvline(union_over_inter_clouds.index(max(union_over_inter_clouds)), color="green", linewidth=0.7, linestyle="--")
    plt.axvline(union_over_inter_water.index(max(union_over_inter_water)), color="red", linewidth=0.7, linestyle="--")

    plt.subplot(2, 3, 4)
    plt.plot(dice_background, label="background")
    plt.plot(dice_snow, label="snow")
    plt.plot(dice_clouds, label="clouds")
    plt.plot(dice_water, label="water")
    plt.title("Dice")
    plt.ylim(0.75, 1)
    plt.legend()

    # mark each maximum level with a thin dashed horizontal line in the corresponding color
    plt.axhline(max(dice_background), color="blue", linewidth=0.7, linestyle="--")
    plt.axhline(max(dice_snow), color="orange", linewidth=0.7, linestyle="--")
    plt.axhline(max(dice_clouds), color="green", linewidth=0.7, linestyle="--")
    plt.axhline(max(dice_water), color="red", linewidth=0.7, linestyle="--")

    # mark each maximum level with a thin dashed vertical line in the corresponding color
    plt.axvline(dice_background.index(max(dice_background)), color="blue", linewidth=0.7, linestyle="--")
    plt.axvline(dice_snow.index(max(dice_snow)), color="orange", linewidth=0.7, linestyle="--")
    plt.axvline(dice_clouds.index(max(dice_clouds)), color="green", linewidth=0.7, linestyle="--")
    plt.axvline(dice_water.index(max(dice_water)), color="red", linewidth=0.7, linestyle="--")

    plt.show()


if __name__ == "__main__":
    main()
