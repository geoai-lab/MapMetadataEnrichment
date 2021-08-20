import argparse
import numpy as np
from utils.openmax import fit_weibull, openmax


def main(Region_ID):
    parser = argparse.ArgumentParser()
    parser.add_argument('--distance_type', default='eucos')
    parser.add_argument('--euc_scale', type=float, default=5e-3)
    parser.add_argument('--alpha', type=int, default=5)
    parser.add_argument('--tailsize', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=0.111) # 1.0 by 9 classes
    parser.add_argument('--save', default='None', help='Directory to save the results')
    parser.add_argument('--score_path', default='None')
    opt = parser.parse_args()

    categories = ["Africa", "Antarctica", "Asia", "Europe", "Global", "North America", "Oceania", "South America"]
    means = np.load("mavs.npy",allow_pickle=True)
    dists = np.load("dists.npy", allow_pickle=True)
    scores_test = np.load("test_scores.npy", allow_pickle=True)[Region_ID]

    weibull_model = fit_weibull(means, dists, categories, opt.tailsize, opt.distance_type)
    print("Finish fitting Weibull distribution!")
    
    pred_y, pred_y_o = [], []
    for score in scores_test:
        so, ss = openmax(weibull_model, categories, score, opt.euc_scale, opt.alpha, opt.distance_type)
        pred_y.append(np.argmax(ss) if np.max(ss) >= opt.threshold else len(categories))
        pred_y_o.append(np.argmax(so) if np.max(so) >= opt.threshold else len(categories))

    print("SoftMax classification accuracy is:")
    print((np.array(pred_y)==Region_ID).sum()/100)
    print("-----------------------------------")
    print("OpenMax classification accuracy is:")
    print((np.array(pred_y_o)==Region_ID).sum()/100)


if __name__ == "__main__":
    classes = ["Africa", "Antarctica", "Asia", "Europe", "Global", "North America", "Oceania", "South America", "Noise"]
    for idx, item in enumerate(classes):
        print("Processing on the map images from the class of " + item)
        main(idx)