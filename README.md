# kmedoids

3 different variations of k medoids are implemented. 

* PAM
* Clara
* PAM-lite

Usage: No constraints on data type. Self-defined distance functions must be provided.

```python
    def disFn(a, b):
        return abs(a - b)

    data = []
    for i in range(50):
        data.append(1 + i)
    for i in range(50):
        data.append(1000 + i)

    k_medoids = KMedoids()

    cost, medoids, clusters = k_medoids.clara(data, 2, disFn)
    # cost, medoids, clusters = k_medoids.pam_lite(data, 2, disFn)
    # cost, medoids, clusters = k_medoids.pam(data, 2, disFn, 1000)
    print('cost', cost)
    print('medoids', medoids)
    print('clusters', clusters)
    print('davies bouldin index', k_medoids.davies_bouldin_score(data, clusters, disFn))
```