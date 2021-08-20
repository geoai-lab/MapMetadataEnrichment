import numpy as np
import scipy.spatial.distance as spd


def compute_channel_distances(mavs, features, eu_weight):
    """
    Input:
        mavs (channel, C)
        features: (N, channel, C)
    Output:
        channel_distances: dict of distance distribution from MAV for each channel.
    """
    eucos_dists, eu_dists, cos_dists = [], [], []
    for channel, mcv in enumerate(mavs):  # Compute channel specific distances
        eu_dists.append([spd.euclidean(mcv, feat[channel]) for feat in features])
        cos_dists.append([spd.cosine(mcv, feat[channel]) for feat in features])
        eucos_dists.append([spd.euclidean(mcv, feat[channel]) * eu_weight +
                            spd.cosine(mcv, feat[channel]) for feat in features])

    return {'eucos': np.array(eucos_dists), 'cosine': np.array(cos_dists), 'euclidean': np.array(eu_dists)}


def main():
    """
    Compute category spesific distance distribution
    """
    scores = np.load("train_scores.npy", allow_pickle=True)
    mavs = np.load("mavs.npy")
    # print()
    dists = [compute_channel_distances(mcv, score, 5e-3) for mcv, score in zip(mavs, scores)]
    # print(dists)
    np.save("dists.npy", dists)


if __name__ == "__main__":
    main()