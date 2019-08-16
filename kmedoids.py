import random


class KMedoids():

    def __init__(self, data, dist_fn, dist_file=''):
        self.__data = data
        self.__dist_fn = dist_fn
        self.__dist_cache = []

        for i in range(len(data)):
            self.__dist_cache.append([None] * (len(data) - i))

        count = 0
        if dist_file != '':
            with open(dist_file, 'r') as f:
                lines = f.readlines(1024000)

                while len(lines) != 0:
                    count += len(lines)
                    for line in lines:
                        parts = line[:-1].split(' ')

                        x = int(parts[0])
                        y = int(parts[1]) - x

                        self.__dist_cache[x][y] = float(parts[2])

                    del lines

                    lines = f.readlines(1024000)

            print('cache_size', len(self.__dist_cache))

    def __get_dist(self, i, j):
        if i == j:
            return 0
        else:
            if i < j:
                x, y = i, j - i
            else:
                x, y = j, i - j

            if self.__dist_cache[x][y] != None:
                return self.__dist_cache[x][y]
            else:
                dist = self.__dist_fn(self.__data[i], self.__data[j])
                # print('cachekey', (x, y), 'does not exist', dist)
                self.__dist_cache[x][y] = dist
                return dist

    def davies_bouldin_score(self, clusters):
        D = 0.0

        for mi, ci in clusters.items():
            max_rij = -float('inf')

            avg_dist_i = 0.0
            for i in ci:
                avg_dist_i += self.__get_dist(i, mi)
            avg_dist_i /= len(ci)

            for mj, cj in clusters.items():
                if mi != mj:
                    avg_dist_j = 0.0
                    for j in cj:
                        avg_dist_j += self.__get_dist(j, mj)
                    avg_dist_j /= len(cj)

                    rij = (avg_dist_i + avg_dist_j) / \
                        self.__get_dist(mi, mj)
                    if rij > max_rij:
                        max_rij = rij

            D += max_rij

        return D / len(clusters)

    def clara(self, k, runs=5, sample_base=40):
        """The main clara clustering iterative algorithm.

        :param k: Number of medoids.
        :return: The minimized cost, the best medoid choices and the final configuration.
        """
        size = len(self.__data)

        min_cost = float('inf')
        best_choices = []
        best_results = {}

        for j in range(runs):

            sampling_idx = random.sample(
                [i for i in range(size)], (sample_base+k*2))

            # print('best choices and sample', best_choices, sampling_idx)

            sampling_data = []
            index_mapping = {}

            for idx in sampling_idx:
                sampling_data.append(self.__data[idx])
                index_mapping[len(sampling_data) - 1] = idx

            kmedoids = KMedoids(sampling_data, self.__dist_fn)

            _, medoids, _ = kmedoids.pam(k)

            # map from sample index to original data set index
            medoids = [index_mapping[i] for i in medoids]

            cost, clusters = self.__form_clusters(medoids)

            if cost <= min_cost:
                min_cost = cost
                best_choices = list(medoids)
                best_results = dict(clusters)

        return best_choices, best_results

    def pam_lite(self, k, runs=5, sample_base=40):
        """A variation of PAM algorithm
        from Olukanmi, P. O., Nelwamondo, F., & Marwala, T. (2019, January). PAM-lite: fast and accurate k-medoids clustering for massive datasets. In 2019 Southern African Universities Power Engineering Conference/Robotics and Mechatronics/Pattern Recognition Association of South Africa (SAUPEC/RobMech/PRASA) (pp. 200-204). IEEE.
        """

        size = len(self.__data)

        D = set()  # a set to hold all medoids of different runs

        for j in range(runs):
            sampling_idx = random.sample(
                [i for i in range(size)], (sample_base+k*2))
            sampling_data = []
            index_mapping = {}

            for idx in sampling_idx:
                sampling_data.append(self.__data[idx])
                index_mapping[len(sampling_data) - 1] = idx

            kmedoids = KMedoids(sampling_data, self.__dist_fn)

            _, medoids, _ = kmedoids.pam(k)

            # map from sample index to original data set index
            medoids = [index_mapping[i] for i in medoids]

            for m in medoids:
                D.add(m)

        D = list(D)
        Dd = []
        index_mapping = {}
        for idx in D:
            Dd.append(self.__data[idx])
            index_mapping[len(Dd) - 1] = idx

        kmedoids = KMedoids(Dd, self.__dist_fn)

        _, medoids, _ = kmedoids.pam(k)

        medoids = [index_mapping[i] for i in medoids]

        cost, clusters = self.__form_clusters(medoids)

        return medoids, clusters

    def pam(self, k, init_medoids=[]):
        """The original Partitioning Around Medoids algorithm.

        :param k: Number of medoids.
        :return: medoids and clusters.
        """
        # print('K-medoids starting')
        # Do some smarter setting of initial cost configuration
        build_medoids = self.__build_phase(k, init_medoids)

        medoids = list(build_medoids)

        print('build phase finished', medoids)

        while True:
            min_result = float('inf')
            swap = ()

            U = [i for i in range(len(self.__data)) if i not in medoids]

            for i in range(len(medoids)):
                for h in U:
                    # print('pam swap', i, h)
                    result = self.__swap_result(medoids[i], h, medoids, U)
                    # print('processing', self.__data[medoids[i]], self.__data[h], [self.__data[t] for t in medoids], [self.__data[t] for t in U], result)
                    if result < min_result:
                        min_result = result
                        swap = (i, h)

            if min_result < 0:
                medoids[swap[0]] = swap[1]
                print('swap', swap, min_result, medoids)

                # prevent possible deadlock
                if min_result > -0.3:
                    break
            else:
                break

        _, clusters = self.__form_clusters(medoids)

        return build_medoids, medoids, clusters

    def clarans(self, k, numlocal, max_neighbour):
        size = len(self.__data)

        current_cost = float('inf')
        current_medoids = random.sample([i for i in range(size)], k)

        best_medoids = current_medoids
        min_cost = float('inf')

        for i in range(numlocal):
            # print('run', i, 'current', current_medoids)

            j = 0
            while j < max_neighbour:
                replace_index = random.randint(0, k - 1)

                U = [i for i in range(len(self.__data))
                     if i not in current_medoids]

                replacement = random.sample(U, 1)[0]

                cost = self.__swap_result(
                    current_medoids[replace_index], replacement, current_medoids, U)
                if cost < 0:
                    current_cost = cost
                    current_medoids[replace_index] = replacement

                    j = 0
                else:
                    j += 1

            if current_cost < min_cost:
                min_cost = current_cost
                best_medoids = current_medoids
                print('run', i, 'best medoids', best_medoids)

            current_cost = float('inf')
            current_medoids = random.sample([i for i in range(size)], k)

        _, clusters = self.__form_clusters(best_medoids)

        return best_medoids, clusters

    def __swap_result(self, i, h, medoids, U):
        # tmp_medoids = [m for m in medoids if m != i]
        tmp_medoids = medoids

        tih = 0

        for j in U:
            if j == h:
                continue

            djh = self.__get_dist(j, h)
            dji = self.__get_dist(j, i)

            min_dist = min(self.__get_dist(
                j, tmp_medoids[0]), self.__get_dist(j, tmp_medoids[1]))
            second_dist = max(self.__get_dist(
                j, tmp_medoids[0]), self.__get_dist(j, tmp_medoids[1]))

            for k in range(2, len(tmp_medoids)):
                dist = self.__get_dist(j, tmp_medoids[k])
                if dist < min_dist:
                    second_dist = min_dist
                    min_dist = dist
                elif dist < second_dist:
                    second_dist = dist

            # print('min second dist', min_dist, second_dist)

            # j is more distant from both i and h than from one of the other representative objects
            if min_dist < djh and min_dist < dji:
                cjih = 0
            # j is not further from i than from any other selected representative object
            elif abs(dji - min_dist) < 1e-6:
                # j is closer to h than to the second closest representative object
                if djh < second_dist:
                    cjih = djh - dji
                # j is at least as distant from h than from the second closest representative object
                else:
                    cjih = second_dist - min_dist
            else:
                cjih = djh - min_dist

            tih += cjih

        # print('tih', i, h, tih)

        return tih

    def __form_clusters(self, _cur_choice):
        """A function to compute the configuration cost.

        :param _cur_choice: The current set of medoid choices.
        :return: The total configuration cost, the medoids.
        """
        size = len(self.__data)
        total_distance = 0.0
        clusters = {}
        for idx in _cur_choice:
            clusters[idx] = [idx]

        for i in range(size):
            if i not in clusters:
                choice = -1
                min_dist = float('inf')

                for m in clusters:
                    tmp = self.__get_dist(m, i)

                    if tmp < min_dist:
                        choice = m
                        min_dist = tmp

                clusters[choice].append(i)
                total_distance += min_dist

        return total_distance, clusters

    def __build_phase(self, k, init_medoids=[]):
        medoids = list(init_medoids)

        if len(medoids) == 0:
            min_dist = float('inf')
            first_medoid = -1

            # compute and add first medoid
            for i in range(len(self.__data)):
                dist = 0

                for j in range(len(self.__data)):
                    if i != j:
                        dist += self.__get_dist(i, j)

                if dist < min_dist:
                    first_medoid = i
                    min_dist = dist

            medoids.append(first_medoid)

        # compute and add remaining medoids
        for n in range(k - len(medoids)):
            max_gi = -float('inf')
            candidate = -1

            for i in range(len(self.__data)):
                if i in medoids:
                    continue

                gi = 0

                for j in range(len(self.__data)):
                    Dj = float('inf')

                    if i != j and j not in medoids:
                        for m in medoids:
                            dist = self.__get_dist(j, m)
                            if dist < Dj:
                                Dj = dist

                        Cji = max(Dj - self.__get_dist(j, i), 0)
                        gi += Cji

                if gi > max_gi:
                    max_gi = gi
                    candidate = i

            medoids.append(candidate)

        return medoids

    def silhouette_scores(self, clusters):
        result = [None] * len(self.__data)

        for c, members in clusters.items():
            for i in members:
                if len(members) == 1:
                    result[i] = 0
                    break

                # calculate intra cluster avg distance
                intra_dist = 0
                for j in members:
                    intra_dist += self.__get_dist(i, j)
                intra_dist /= len(members) - 1

                min_inter_dist = float('inf')
                closest = -1

                # calculate min inter cluster avg distance
                for d, points in clusters.items():
                    if c == d:
                        continue

                    inter_dist = 0
                    for p in points:
                        inter_dist += self.__get_dist(i, p)
                    inter_dist /= len(points)

                    if inter_dist < min_inter_dist:
                        min_inter_dist = inter_dist
                        closest = d

                score = (min_inter_dist - intra_dist) / \
                    max(min_inter_dist, intra_dist)
                result[i] = score

        return result


if __name__ == '__main__':
    def disFn(a, b):
        return abs(a - b)

    data = []
    for i in range(49):
        data.append(0 + i)
    for i in range(49):
        data.append(1000 + i)

    random.shuffle(data)

    print(data)

    k_medoids = KMedoids(data, disFn)

    _, medoids, clusters = k_medoids.pam(2)
    # medoids, clusters = k_medoids.clara(2)
    # medoids, clusters = k_medoids.pam_lite(2)
    # medoids, clusters = k_medoids.clarans(2, 20, 80)

    print('medoids', medoids, [data[i] for i in medoids])
    # print('clusters', clusters)
    for k, v in clusters.items():
        print('cluster', data[k], [data[i] for i in v])
    # print('davies bouldin index',
    #       k_medoids.davies_bouldin_score(clusters))
