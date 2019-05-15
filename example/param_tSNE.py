# -*- coding: utf-8 -*-
from __future__ import division  # Python 2 users only
from __future__ import print_function

__doc__ = """ usage of parametric_tSNE."""

import sys
import datetime
import os
import numpy as np
import time

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import ConfigParser

plt.style.use('ggplot')
from matplotlib.backends.backend_pdf import PdfPages

cur_path = os.path.realpath(__file__)
_cur_dir = os.path.dirname(cur_path)
_par_dir = os.path.abspath(os.path.join(_cur_dir, os.pardir))
sys.path.append(_cur_dir)
sys.path.append(_par_dir)
from parametric_tSNE import Parametric_tSNE
from parametric_tSNE.utils import get_multiscale_perplexities

has_sklearn = False
try:
    from sklearn.decomposition import PCA

    has_sklearn = True
except Exception as ex:
    print('Error trying to import sklearn, will not plot PCA')
    print(ex)
    pass

def _plot_scatter(output_res, pick_rows, std_label, color_palette, symbols, alpha=0.5):
    symcount = len(symbols)
    for idx, alphabet in enumerate(std_label):
        cur_plot_rows = pick_rows == alphabet
        cur_color = color_palette[idx]
        plt.plot(output_res[cur_plot_rows, 0], output_res[cur_plot_rows, 1], marker= symbols[idx%symcount],
                 color=cur_color, label=alphabet, alpha=alpha)


def _plot_kde(output_res, pick_rows, std_label, color_palette, alpha=0.5):
    for idx, alphabet in enumerate(std_label):
        cur_plot_rows = pick_rows == alphabet
        cur_cmap = sns.light_palette(color_palette[idx], as_cmap=True)
        sns.kdeplot(output_res[cur_plot_rows, 0], output_res[cur_plot_rows, 1], cmap=cur_cmap, shade=True, alpha=alpha,
                    shade_lowest=False)
        centroid = output_res[cur_plot_rows, :].mean(axis=0)
        plt.annotate(alphabet, xy=centroid, xycoords='data', alpha=alpha,
                     horizontalalignment='center', verticalalignment='center')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('set config file')

    # load config file.
    config = ConfigParser.ConfigParser()
    if config.read(sys.argv[1]) == []:
        print("There is no config file: " + sys.argv[1])
        exit(-1)

    dimension = config.getint('tSNE_param', 'input_dimension')
    out_dim = config.getint('tSNE_param', 'output_dimension')
    perplexity = config.getint('tSNE_param', 'perplexity')
    alpha_ = config.getfloat('tSNE_param', 'alpha')
    batch_size = config.getint('tSNE_param', 'batch_size')
    epochs = config.getint('tSNE_param', 'epochs')
    do_pretrain = config.getboolean('tSNE_param', 'do_pretrain')
    random_seed = config.getint('tSNE_param', 'random_seed')

    train_file = config.get('data', 'train_file')
    test1_file = config.get('data', 'test1_file')
    test2_file = config.get('data', 'test2_file')

    outfile_frame = config.get('visual', 'out_file')


    font_location = 'NanumGothic.ttf'
    font_name = fm.FontProperties(fname = font_location).get_name()
    mpl.rc('font', family=font_name)

    symbollist = ['o', 'x', '+', 'v', '^', '<', '>', '*']

    startTime = time.time()

    print('loading data...')

    GT = np.loadtxt('HanSeq.csv', delimiter=',', dtype=np.unicode)
    num_cluster = len(GT)
    color_palette = sns.color_palette("hls", num_cluster)

    colslist = [i for i in range(dimension)]
    colstuple = tuple(colslist)

    train_data = np.loadtxt(train_file, delimiter=',', dtype=np.float32, usecols=colstuple, encoding='utf-8-sig')
    test1_data = np.loadtxt(test1_file, delimiter=',', dtype=np.float32, usecols=colstuple, encoding='utf-8-sig')
    test2_data = np.loadtxt(test2_file, delimiter=',', dtype=np.float32, usecols=colstuple, encoding='utf-8-sig')

    train_label = np.loadtxt(train_file, delimiter=',', dtype=np.unicode, usecols={63}, encoding='utf-8-sig')
    test1_label = np.loadtxt(test1_file, delimiter=',', dtype=np.unicode, usecols={63}, encoding='utf-8-sig')
    test2_label = np.loadtxt(test2_file, delimiter=',', dtype=np.unicode, usecols={63}, encoding='utf-8-sig')

    print('data loaded. elapsed time = {}'.format(time.time() - startTime))


    label_list = [os.path.splitext(train_file)[0], os.path.splitext(test1_file)[0], os.path.splitext(test2_file)[0]]

    transformer_list = [{'title': os.path.splitext(train_file)[0], 'data': train_data, 'label': train_label},
                        {'title': os.path.splitext(test1_file)[0], 'data': test1_data, 'label': test1_label},
                        {'title': os.path.splitext(test2_file)[0], 'data': test2_data, 'label': test2_label}]

    print('tSNE train start...')
    ptSNE = Parametric_tSNE(dimension, out_dim, perplexity,  alpha=alpha_, do_pretrain=do_pretrain, batch_size=batch_size, seed=random_seed )
    ptSNE.fit(train_data, epochs=epochs,verbose=1)
    train_result = ptSNE.transform(train_data)

    pdf_obj = PdfPages(outfile_frame.format(perp_tag = perplexity))

    for idx, tlist in enumerate(transformer_list):
        test_result = ptSNE.transform(tlist['data'])

        plt.figure()
        # Create a contour plot of training data
        _plot_kde(test_result, tlist['label'], GT, color_palette, 0.5)
        #_plot_kde(train_result, train_label, GT, color_palette, 1.0)

        # Scatter plot of test data
        _plot_scatter(test_result, tlist['label'], GT, color_palette, symbols=symbollist, alpha=0.1)

        leg = plt.legend(bbox_to_anchor=(1.1, 1.0), fontsize='small')
        #Set marker to be fully opaque in legend
        for lh in leg.legendHandles:
            lh._legmarker.set_alpha(1.0)

        plt.title('{title_tag}_Perplexity({perp_tag})'.format(title_tag = tlist['title'], perp_tag = perplexity))
        plt.savefig(pdf_obj, format='pdf')

    pdf_obj.close()


    print('elased Time = {}'.format(time.time() - startTime))
