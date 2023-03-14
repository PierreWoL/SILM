
from scipy.integrate import quad as integrate

class LSHAbstract:
    @staticmethod
    def _false_positive_probability(threshold, b, r):
        """
               Computes the probability of false positive occurrence in LSH.
               Parameters
               ----------
               threshold : float
                   The minimum similarity threshold.
               b : int
                   The number of bands used with LSH.
                   b * r = the size of the underlying hash.
               r : int
                   The number of rows in each band.
                   b * r = the size of the underlying hash.
               Returns
               -------
               float
                   The probability of false positive occurrence
               """
        _probability = lambda s: 1 - (1 - s ** float(r)) ** float(b)
        a, err = integrate(_probability, 0.0, threshold)[:1]
        return a

    @staticmethod
    def _false_negative_probability(threshold, b, r):
        _probability = lambda s: 1 - (1 - (1 - s ** float(r)) ** float(b))
        a, err = integrate(_probability, threshold, 1.0)[:1]
        return a

    def optimal_param(self, threshold, num_perm, false_positive_weight,
                      false_negative_weight):
        '''
        Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
        of probabilities of false positive and false negative.
        '''
        min_error = float("inf")
        opt = (0, 0)
        for b in range(1, num_perm + 1):
            max_r = int(num_perm / b)
            for r in range(1, max_r + 1):
                fp = self._false_positive_probability(threshold, b, r)
                fn = self._false_negative_probability(threshold, b, r)
                error = fp * false_positive_weight + fn * false_negative_weight
                if error < min_error:
                    min_error = error
                    opt = (b, r)
        return opt
