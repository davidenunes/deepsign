from deepsign.utils.profiling import total_size

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt


# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

def config_plot(max_sentences):



    # Remove the plot frame lines. They are unnecessary chartjunk.
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.set_xlim(0, max_sentences)




    # Limit the range of the plot to only where the data is.
    # Avoid unnecessary whitespace.
    #plt.ylim(0, 90)
    #plt.xlim(1968, 2014)

    # Make sure your axis ticks are large enough to be easily read.
    # You don't want your viewers squinting to read your plot.
    #plt.yticks(range(0, 91, 10), [str(x) + "%" for x in range(0, 91, 10)], fontsize=14)
    plt.xticks(fontsize=14)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])


plt.ion()
for i in range(1000):
    pass #do stuff


if i % 10000 == 0:
    #word_ids = [w_id for w_id in sign_index.signs.values()]
    #words = [sign_index.get_sign(w_id) for w_id in word_ids]
    #ri = [ri_w.positive + ri_w.negative for ri_w in sign_index.random_indexes.values()]

    words = []
    word_ids = []
    ri = []

    word_size = total_size(words) / pow(2, 20)
    word_id_size = total_size(word_ids * 2) / pow(2, 20)
    ri_size = total_size(ri) / pow(2, 20)

    # tqdm.write("len index: {} words".format(len(sign_index)))
    # tqdm.write("word ids memory footprint: {0:.3f}MB".format(round(word_id_size, 3)))
    # tqdm.write("word memory footprint: {0:.3f}MB".format(round(word_size, 3)))
    # tqdm.write("random index memory footprint: {0:.3f}MB".format(round(ri_size, 3)))

    word_plot = plt.scatter(i, word_size, label="words")
    plt.setp(word_plot, color=tableau20[0])

    ids_plot = plt.scatter(i, word_id_size, label="ids")
    plt.setp(ids_plot, color=tableau20[1])

    ri_plot = plt.scatter(i, ri_size, label="ri")
    plt.setp(ri_plot, color=tableau20[2])

    plt.legend(handles=[word_plot, ids_plot, ri_plot], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel('Memory Usage in MB')

    plt.ylim(ymin=0, auto=True)

    plt.pause(0.001)