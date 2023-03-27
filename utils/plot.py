import pandas as pd
import matplotlib.pyplot as plt
import glob


def plotRatingHist(column, x_lim, y_lim, title="", save=False):
    column.hist(bins=30)
    if x_lim and y_lim:
        ax = plt.gca()
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
    plt.suptitle(title)
    if save:
        plt.savefig("../plots/hist_" + title)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    folderPath = "../data/df_genres/*"
    for file in glob.glob(folderPath):
        data = pd.read_csv(file)
        genreName = file.split("/")[-1].split("df_")[-1].split(".csv")[0]
        plotRatingHist(data.ratings, x_lim=[0, 5], y_lim=[0, 4200], title="Ratings_" + genreName, save=True)
