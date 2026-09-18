import math
import itertools
import numpy as np


def PE_func(timeSeries, tauList, D):
    """
    Python translation of the MATLAB function PE_func.

    Parameters
    ----------
    timeSeries : array-like
        Input time series.
    tauList : array-like
        List/array of integer delays tau.
    D : int
        Embedding dimension.

    Returns
    -------
    saida : numpy.ndarray, shape (len(tauList), 3)
        Columns: [normalized Shannon entropy H,
                  Jensen-Shannon statistical complexity C,
                  Fisher information F].
    """

    # MATLAB's length() treats a row/column vector as its number of elements.
    timeSeries = np.asarray(timeSeries).reshape(-1)
    tauList = np.asarray(tauList).reshape(-1)

    saida = np.zeros((len(tauList), 3), dtype=float)

    N = math.factorial(D)       # Number of possible ordinal patterns
    S_max = np.log2(N)          # Maximum entropy

    #   >>>>>   SYMBOL/LEHMER   <<<<<
    # MATLAB: all_symbols = flipud(perms(1:D));
    # flipud(perms(1:D)) gives the permutations in lexicographic ascending
    # order, which is exactly the order generated here by itertools.permutations.
    all_symbols = np.array(
        list(itertools.permutations(range(1, D + 1))),
        dtype=int,
    )

    # MATLAB: powers = 10.^(D-1:-1:0);
    powers = 10 ** np.arange(D - 1, -1, -1, dtype=np.int64)

    # MATLAB: all_symbols_flat = sum(all_symbols .* repmat(powers,[N,1]),2);
    all_symbols_flat = np.sum(all_symbols * powers, axis=1).astype(np.int64)

    # MATLAB uses the numeric symbol itself as a 1-based array index.
    # We allocate one extra position so Python can use that same symbol directly.
    index_map = np.zeros(int(np.max(all_symbols_flat)) + 1, dtype=np.int64)

    # Store MATLAB-style Bandt-Pompe indices 1..N.
    for i in range(N):
        symbol = all_symbols_flat[i]
        index_map[symbol] = i + 1

    for tauId, tau in enumerate(tauList):
        tau = int(tau)

        # Compute the Bandt-Pompe probability distribution.
        P = PDF_modification_allelements(
            timeSeries, D, tau, 1, index_map, powers, N
        )

        # Compute the normalized Shannon entropy.
        H = Shannon(P) / S_max

        # Compute the Jensen-Shannon statistical complexity.
        C = Complexity_Jensen_Shannon(P, N, H)

        # Compute the Fisher information measure.
        F = Fisher(P)

        # Store entropy, complexity, and Fisher information for the current tau.
        saida[tauId, :] = [H, C, F]

    return saida


#   >>>>>   BANDT-POMPE PDF   <<<<<
def PDF_modification_allelements(
    timeSeriesOriginal, D, tau, direction, index_map, powers, N
):
    P = np.zeros(N, dtype=float)

    # MATLAB: for k = 1:tau
    for k in range(1, tau + 1):

        # MATLAB: timeSeriesOriginal(k:tau:end)
        # MATLAB index k corresponds to Python index k-1.
        timeSeries = timeSeriesOriginal[k - 1 :: tau]

        # Reverse the time series when a negative direction is requested.
        if direction < 0:
            timeSeries = np.flipud(timeSeries)

        T = len(timeSeries)
        nWin = T - D + 1

        # Skip this phase if there are not enough samples to form a window.
        if nWin <= 0:
            continue

        # Create all embedding windows at once.
        X = np.zeros((nWin, D), dtype=timeSeries.dtype)

        # MATLAB:
        # for r = 1:D
        #     X(:, r) = timeSeries(r:r+nWin-1);
        # end
        for r in range(D):
            X[:, r] = timeSeries[r : r + nWin]

        # MATLAB: [~, order] = sort(X, 2, 'ascend');
        # +1 converts Python positions 0..D-1 into MATLAB-style positions 1..D.
        # Stable sorting preserves the original order when equal values occur.
        order = np.argsort(X, axis=1, kind="stable") + 1

        # MATLAB: symbols = order * transpose(powers);
        symbols = (order @ powers).astype(np.int64)

        # MATLAB: lehmer_indices = index_map(symbols);
        # Values stored in index_map are still MATLAB-style indices 1..N.
        lehmer_indices = index_map[symbols]

        # MATLAB: accumarray(lehmer_indices, 1, [N, 1])
        # np.bincount is zero-based, hence lehmer_indices - 1.
        P += np.bincount(lehmer_indices - 1, minlength=N)[:N]

    # MATLAB: P = P ./ sum(P);
    total = np.sum(P)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = P / total

    return P


#   >>>>>   SHANNON ENTROPY   <<<<<
def Shannon(P):
    S = 0.0

    for i in range(len(P)):
        # Ignore zero-probability states to avoid log2(0).
        if P[i] == 0:
            continue

        S = S - P[i] * np.log2(P[i])

    return S


#   >>>>>   JENSEN-SHANNON COMPLEXITY   <<<<<
def Complexity_Jensen_Shannon(P, N, H):
    # Uniform probability distribution.
    P_e = (1 / N) * np.ones(N, dtype=float)

    # Delta distribution used for normalization.
    P_m = np.zeros(N, dtype=float)
    P_m[0] = 1

    # Jensen-Shannon divergence between two probability distributions.
    def Div_Jensen_Shannon(P1, P2):
        return Shannon((P1 + P2) / 2) - Shannon(P1) / 2 - Shannon(P2) / 2

    # Normalization constant.
    Q0 = Div_Jensen_Shannon(P_m, P_e) ** (-1)

    # Normalized disequilibrium.
    Q = Q0 * Div_Jensen_Shannon(P, P_e)

    # Statistical complexity.
    C = Q * H

    return C


#   >>>>>   FISHER INFORMATION   <<<<<
def Fisher(P):
    N = len(P)
    S = 0.0

    # Compute the discrete Fisher information.
    for i in range(N - 1):
        S = S + (np.sqrt(P[i + 1]) - np.sqrt(P[i])) ** 2

    # Use a different normalization factor for extreme delta distributions.
    if ((P[0] == 1 and np.sum(P[1:N]) == 0) or
            (P[N - 1] == 1 and np.sum(P[0:N - 1]) == 0)):
        F0 = 1
    else:
        F0 = 0.5

    S = F0 * S

    return S
