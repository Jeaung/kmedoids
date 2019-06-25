import random


class KMedoids():

    def __init__(self, cache_size=0):
        """
        Parameters
        ----------
        cache_size: the size of distance cache, only values >= 0 are accepted. Default value is 0 which means
        no memory limit is put on the size of cache.
        """
        self.__cache_size = cache_size
        self.__dist_cache = {}

    def davies_bouldin_score(self, X, clusters, _fn):
        """Computes the Davies-Bouldin score.
        The score is defined as the average similarity measure of each cluster with
        its most similar cluster, where similarity is the ratio of within-cluster
        distances to between-cluster distances. Thus, clusters which are farther
        apart and less dispersed will result in a better score.
        The minimum score is zero, with lower values indicating better clustering.
        Read more in the :ref:`User Guide <davies-bouldin_index>`.
        Parameters
        ----------
        X : array-like, shape (``n_samples``, ``n_features``)
            List of ``n_features``-dimensional data points. Each row corresponds
            to a single data point.
        labels : array-like, shape (``n_samples``,)
            Predicted labels for each sample.
        Returns
        -------
        score: float
            The resulting Davies-Bouldin score.
        References
        ----------
        .. [1] Davies, David L.; Bouldin, Donald W. (1979).
        `"A Cluster Separation Measure"
        <https://ieeexplore.ieee.org/document/4766909>`__.
        IEEE Transactions on Pattern Analysis and Machine Intelligence.
        PAMI-1 (2): 224-227
        """
        D = 0.0

        for mi, ci in clusters.items():
            max_rij = -float('inf')

            for mj, cj in clusters.items():
                if mi != mj:
                    avg_dist_i = 0.0
                    for i in ci:
                        avg_dist_i += _fn(X[i], X[mi])
                    avg_dist_i /= len(ci)

                    avg_dist_j = 0.0
                    for j in cj:
                        avg_dist_j += _fn(X[j], X[mj])
                    avg_dist_j /= len(cj)

                    rij = (avg_dist_i + avg_dist_j) / _fn(X[mi], X[mj])
                    if rij > max_rij:
                        max_rij = rij

            D += max_rij

        return D / len(clusters)

    def clara(self, _df, _k, _fn, runs=5, sample_base=40):
        """The main clara clustering iterative algorithm.

        :param _df: Input data frame.
        :param _k: Number of medoids.
        :param _fn: The distance function to use.
        :return: The minimized cost, the best medoid choices and the final configuration.
        """
        size = len(_df)
        niter = 1000

        min_cost = float('inf')
        best_choices = []
        best_results = {}

        for j in range(runs):
            sampling_idx = random.sample(
                [i for i in range(size)], (sample_base+_k*2))
            sampling_data = []
            index_mapping = {}

            for idx in sampling_idx:
                sampling_data.append(_df[idx])
                index_mapping[len(sampling_data) - 1] = idx

            _, medoids, _ = self.pam(sampling_data, _k, _fn, niter)

            # map from sample index to original data set index
            medoids = [index_mapping[i] for i in medoids]

            cost, clusters = self.__compute_cost(_df, _fn, medoids)

            if cost <= min_cost:
                min_cost = cost
                best_choices = list(medoids)
                best_results = dict(clusters)

        return min_cost, best_choices, best_results

    def pam_lite(self, _df, _k, _fn, runs=5, sample_base=40):
        """A variation of PAM algorithm
        from Olukanmi, P. O., Nelwamondo, F., & Marwala, T. (2019, January). PAM-lite: fast and accurate k-medoids clustering for massive datasets. In 2019 Southern African Universities Power Engineering Conference/Robotics and Mechatronics/Pattern Recognition Association of South Africa (SAUPEC/RobMech/PRASA) (pp. 200-204). IEEE.
        """

        size = len(_df)
        niter = 1000

        D = set()  # a set to hold all medoids of different runs

        for j in range(runs):
            sampling_idx = random.sample(
                [i for i in range(size)], (sample_base+_k*2))
            sampling_data = []
            index_mapping = {}

            for idx in sampling_idx:
                sampling_data.append(_df[idx])
                index_mapping[len(sampling_data) - 1] = idx

            _, medoids, _ = self.pam(sampling_data, _k, _fn, niter)

            # map from sample index to original data set index
            medoids = [index_mapping[i] for i in medoids]

            for m in medoids:
                D.add(m)

        D = list(D)
        Dd = []
        index_mapping = {}
        for idx in D:
            Dd.append(_df[idx])
            index_mapping[len(Dd) - 1] = idx

        _, medoids, _ = self.pam(Dd, _k, _fn, niter)

        medoids = [index_mapping[i] for i in medoids]

        cost, clusters = self.__compute_cost(_df, _fn, medoids)

        return cost, medoids, clusters

    def pam(self, _df, _k, _fn, _niter):
        """The original k-mediods algorithm.

        :param _df: Input data frame.
        :param _k: Number of medoids.
        :param _fn: The distance function to use.
        :param _niter: The number of iterations.
        :return: Cluster label.

        Pseudo-code for the k-mediods algorithm.
        1. Sample k of the n data points as the medoids.
        2. Associate each data point to the closest medoid.
        3. While the cost of the data point space configuration is decreasing.
            1. For each medoid m and each non-medoid point o:
                1. Swap m and o, recompute cost.
                2. If global cost increased, swap back.
        """
        print('K-medoids starting')
        # Do some smarter setting of initial cost configuration
        pc1, medoids = self.__cheat_at_sampling(_df, _k, _fn, 17)
        prior_cost, clusters = self.__compute_cost(
            _df, _fn, medoids, use_cache=True)
        # print('init medoids', _df, medoids, [_df[i] for i in medoids], prior_cost)

        current_cost = prior_cost
        iter_count = 0
        best_medoids = medoids
        best_clusters = clusters

        print('Running with {m} iterations'.format(m=_niter))

        while iter_count < _niter:
            for idx in range(len(medoids)):
                for itemIdx in range(len(_df)):
                    if itemIdx not in medoids:
                        swap_temp = medoids[idx]
                        medoids[idx] = itemIdx

                        # print('swap', swap_temp, itemIdx)

                        tmp_cost, tmp_clusters = self.__compute_cost(
                            _df, _fn, medoids, use_cache=True)

                        if tmp_cost < current_cost:
                            # print('cost decreases', iter_count, medoids, itemIdx, tmp_clusters, tmp_cost)
                            best_medoids = list(medoids)
                            best_clusters = tmp_clusters
                            current_cost = tmp_cost
                        else:
                            medoids[idx] = swap_temp

            iter_count += 1

            if abs(current_cost - prior_cost) < 0.0001:
                break
            else:
                prior_cost = current_cost
                # print('best choices', best_choices)

        return current_cost, best_medoids, best_clusters

    def __compute_cost(self, _df, _fn, _cur_choice, use_cache=False):
        """A function to compute the configuration cost.

        :param _df: The input data frame.
        :param _fn: The distance function.
        :param _cur_choice: The current set of medoid choices.
        :return: The total configuration cost, the mediods.
        """
        size = len(_df)
        total_cost = 0.0
        clusters = {}
        for idx in _cur_choice:
            clusters[idx] = []

        for i in range(size):
            choice = -1
            min_cost = float('inf')

            for m in clusters:
                if m == i:
                    tmp = 0
                elif use_cache:
                    if m < i:
                        cache_key = (m, i)
                    else:
                        cache_key = (i, m)

                    if cache_key not in self.__dist_cache:
                        # print('cache misses', cache_key)
                        tmp = _fn(_df[m], _df[i])
                        if self.__cache_size == 0 or len(self.__dist_cache[cache_key]) < self.__cache_size:
                            self.__dist_cache[cache_key] = tmp
                    else:
                        # print('cache hits', cache_key)
                        tmp = self.__dist_cache[cache_key]
                else:
                    tmp = _fn(_df[m], _df[i])

                if tmp < min_cost:
                    choice = m
                    min_cost = tmp

            clusters[choice].append(i)
            total_cost += min_cost

        return total_cost, clusters

    def __cheat_at_sampling(self, _df, _k, _fn, _nsamp):
        """A function to cheat at sampling for speed ups.

        :param _df: The input data frame.
        :param _k: The number of mediods.
        :param _fn: The distance function.
        :param _nsamp: The number of samples.
        :return: The best score, the medoids.
        """
        size = len(_df)
        score_holder = []
        medoid_holder = []
        for _ in range(_nsamp):
            medoids_sample = random.sample([i for i in range(size)], _k)
            cost, medoids = self.__compute_cost(
                _df, _fn, medoids_sample)
            score_holder.append(cost)
            medoid_holder.append(medoids)

        idx = score_holder.index(min(score_holder))
        ms = list(medoid_holder[idx].keys())
        return score_holder[idx], ms


if __name__ == '__main__':
    def disFn(a, b):
        return abs(a - b)

    data = []
    for i in range(50):
        data.append(1 + i)
    for i in range(50):
        data.append(1000 + i)

    k_medoids = KMedoids()

    # cost, medoids, clusters = k_medoids.clara(data, 2, disFn)
    # cost, medoids, clusters = k_medoids.pam_lite(data, 2, disFn)
    cost, medoids, clusters = k_medoids.pam(data, 2, disFn, 1000)
    print('cost', cost)
    print('medoids', medoids)
    print('clusters', clusters)
    print('davies bouldin index',
          k_medoids.davies_bouldin_score(data, clusters, disFn))
