import numpy as np


def decision_tree_regressor(
    X_train, y_train, X_test, max_depth=2, min_samples_split=2
):
    """
    Build a decision tree for regression and predict on test data.
    """

    class RegressionTree:
        def __init__(self, max_depth=max_depth, min_samples_split=min_samples_split):
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.max_nodes = 2 ** (max_depth + 1) - 1

        def fit(self, X, y):
            self.X = np.asarray(X, dtype=float)
            self.y = np.asarray(y, dtype=float).ravel()

            n_samples, n_features = self.X.shape
            M = self.max_nodes

            self.left = np.full(M, -1, dtype=int)
            self.right = np.full(M, -1, dtype=int)
            self.feature = np.full(M, -1, dtype=int)
            self.threshold = np.full(M, np.nan)
            self.value = np.full(M, np.nan)
            self.is_leaf = np.ones(M, dtype=bool)

            self.n_nodes = 1

            stack = [(0, np.arange(n_samples), 0)]

            while stack:
                node, idx, depth = stack.pop()
                self._grow(node, idx, depth, stack)

            return self

        def _grow(self, node, idx, depth, stack):
            y_sub = self.y[idx]
            n = len(idx)

            parent_sse = np.sum((y_sub - y_sub.mean()) ** 2)

            # Stop conditions
            if (
                depth >= self.max_depth
                or n < self.min_samples_split
                or parent_sse <= 1e-12
            ):
                self.value[node] = np.mean(y_sub)
                return

            feat, thr, gain = self._best_split(idx, parent_sse)

            if feat is None or gain <= 1e-12:
                self.value[node] = np.mean(y_sub)
                return

            mask = self.X[idx, feat] <= thr
            idx_L = idx[mask]
            idx_R = idx[~mask]

            if len(idx_L) == 0 or len(idx_R) == 0:
                self.value[node] = np.mean(y_sub)
                return

            self.feature[node] = feat
            self.threshold[node] = thr
            self.is_leaf[node] = False

            left_id = self.n_nodes
            self.n_nodes += 1

            right_id = self.n_nodes
            self.n_nodes += 1

            self.left[node] = left_id
            self.right[node] = right_id

            stack.append((left_id, idx_L, depth + 1))
            stack.append((right_id, idx_R, depth + 1))

        def _best_split(self, idx, parent_sse):
            n_features = self.X.shape[1]
            n_samples = len(idx)

            best_feat = None
            best_thr = 0.0
            best_gain = 0.0

            for i in range(n_features):
                x = self.X[idx, i]
                order = np.argsort(x)

                feat_sort = x[order]
                y_sort = self.y[idx][order]

                for j in range(n_samples - 1):
                    # skip duplicate feature values
                    if feat_sort[j] == feat_sort[j + 1]:
                        continue

                    y_L = y_sort[: j + 1]
                    y_R = y_sort[j + 1 :]

                    n_L = len(y_L)
                    n_R = len(y_R)

                    sse_L = np.sum(y_L**2) - np.sum(y_L) ** 2 / n_L
                    sse_R = np.sum(y_R**2) - np.sum(y_R) ** 2 / n_R

                    thr_j = (feat_sort[j] + feat_sort[j + 1]) / 2.0
                    gain_j = parent_sse - sse_L - sse_R

                    if best_feat is None or gain_j > best_gain:
                        best_feat = i
                        best_thr = thr_j
                        best_gain = gain_j

            return best_feat, best_thr, best_gain

        def predict(self, X):
            X = np.asarray(X, dtype=float)

            node = np.zeros(X.shape[0], dtype=int)

            while True:
                active = ~self.is_leaf[node]
                if not active.any():
                    break

                a = np.where(active)[0]
                cur = node[a]

                go_left = X[a, self.feature[cur]] <= self.threshold[cur]
                node[a] = np.where(
                    go_left,
                    self.left[cur],
                    self.right[cur],
                )

            return self.value[node]

    reg_tree = RegressionTree(max_depth, min_samples_split)
    reg_tree.fit(X_train, y_train)

    y_predict = reg_tree.predict(X_test)

    return np.round(y_predict, 4).tolist()