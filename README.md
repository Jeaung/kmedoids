# kmedoids

4 different variations of k-medoids algorithm are implemented according to their original papers.

* PAM
* Clara
* Clarans
* PAM-lite

Usage: No constraints on data type. Self-defined distance functions must be provided.

```python
    def disFn(a, b):
        return abs(a - b)

    data = []
    for i in range(49):
        data.append(1 + i)
    for i in range(49):
        data.append(1000 + i)

    k_medoids = KMedoids(data, disFn)

    medoids, clusters = k_medoids.pam(2)
    # medoids, clusters = k_medoids.clara(2)
    # medoids, clusters = k_medoids.pam_lite(2)
    # medoids, clusters = k_medoids.clarans(2, 20, 80)

    print('medoids', medoids)
    print('clusters', clusters)
    print('davies bouldin index', k_medoids.davies_bouldin_score(clusters))
```