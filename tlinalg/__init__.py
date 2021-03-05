import numpy as np


def t_product(A, B):
    n_1, n_2, n_3 = A.shape
    m_1, m_2, m_3 = B.shape
    assert n_2 == m_1 and n_3 == m_3, "Inner tensor dimensions must agree."
    A = np.fft.fft(A)
    B = np.fft.fft(B)
    C = np.zeros((n_1, m_2, n_3), dtype=complex)

    C[:, :, 0] = np.dot(A[:, :, 0], B[:, :, 0])
    half_n_3 = round(n_3 / 2)
    for i in range(1, half_n_3):
        C[:, :, i] = np.dot(A[:, :, i], B[:, :, i])
        C[:, :, n_3 - i] = np.conj(C[:, :, i])
    if n_3 % 2 == 0:
        i = half_n_3
        C[:, :, i] = np.dot(A[:, :, i], B[:, :, i])
    C = np.fft.ifft(C).real
    return C


def t_eye(n, n_3):
    I = np.zeros((n, n, n_3))
    I[:, :, 0] = np.eye(n)
    return I


def t_transpose(A):
    n_1, n_2, n_3 = A.shape
    A_t = np.zeros((n_2, n_1, n_3))
    A_t[:, :, 0] = np.conj(A[:, :, 0].T)
    for i in range(1, n_3):
        A_t[:, :, i] = np.conj(A[:, :, n_3 - i].T)
    return A_t


def t_svd(A, opt='full'):
    assert opt in ['full', 'econ', 'skinny'], "opt argument should be one of 'full', 'econ' or 'skinny'"
    n_1, n_2, n_3 = A.shape
    A = np.fft.fft(A)
    if opt == 'full':
        U, S, V = np.zeros((n_1, n_1, n_3), dtype=complex), np.zeros((n_1, n_2, n_3), dtype=complex), np.zeros(
            (n_2, n_2, n_3), dtype=complex)
        u, s, v_h = np.linalg.svd(A[:, :, 0])
        U[:, :, 0], V[:, :, 0] = u, np.conj(v_h.T)
        np.fill_diagonal(S[:, :, 0], s)
        half_n_3 = round(n_3 / 2)
        for i in range(1, half_n_3):
            u, s, v_h = np.linalg.svd(A[:, :, i])
            U[:, :, i], V[:, :, i] = u, np.conj(v_h.T)
            np.fill_diagonal(S[:, :, i], s)
            U[:, :, n_3 - i] = np.conj(U[:, :, i])
            V[:, :, n_3 - i] = np.conj(V[:, :, i])
            S[:, :, n_3 - i] = S[:, :, i]
        if n_3 % 2 == 0:
            i = half_n_3
            u, s, v_h = np.linalg.svd(A[:, :, i])
            U[:, :, i], V[:, :, i] = u, np.conj(v_h.T)
            np.fill_diagonal(S[:, :, i], s)
    elif opt == 'econ' or 'skinny':
        n_min = min(n_1, n_2)
        U, S, V = np.zeros((n_1, n_min, n_3), dtype=complex), np.zeros((n_min, n_min, n_3), dtype=complex), np.zeros(
            (n_2, n_min, n_3), dtype=complex)
        u, s, v_h = np.linalg.svd(A[:, :, 0], full_matrices=False)
        U[:, :, 0], V[:, :, 0] = u, np.conj(v_h.T)
        half_n_3 = round(n_3 / 2)
        for i in range(1, half_n_3):
            u, s, v_h = np.linalg.svd(A[:, :, i], full_matrices=False)
            U[:, :, i], V[:, :, i] = u, np.conj(v_h.T)
            np.fill_diagonal(S[:, :, i], s)
            U[:, :, n_3 - i] = np.conj(U[:, :, i])
            V[:, :, n_3 - i] = np.conj(V[:, :, i])
            S[:, :, n_3 - i] = S[:, :, i]
        if n_3 % 2 == 0:
            i = half_n_3
            u, s, v_h = np.linalg.svd(A[:, :, i], full_matrices=False)
            U[:, :, i], V[:, :, i] = u, np.conj(v_h.T)
            np.fill_diagonal(S[:, :, i], s)
        if opt == 'skinny':
            s_1 = np.diag(np.sum(S, axis=-1)) / n_3
            t_rank = np.sum(s_1 > 1e-10)
            print(t_rank, s_1)
            U = U[:, :t_rank, :]
            V = V[:, :t_rank, :]
            S = S[:t_rank, :t_rank, :]
    U, S, V = np.fft.ifft(U).real, np.fft.ifft(S).real, np.fft.ifft(V).real
    return U, S, V


def t_inv(A):
    n_1, n_2, n_3 = A.shape
    assert n_1 == n_2, "Tensor must be square."
    A = np.fft.fft(A)
    inv_A = np.zeros((n_1, n_2, n_3), dtype=complex)
    inv_A[:, :, 0] = np.linalg.inv(A[:, :, 0])
    half_n_3 = round(n_3 / 2)
    for i in range(1, half_n_3):
        inv_A[:, :, i] = np.linalg.inv(A[:, :, i])
        inv_A[:, :, n_3 - i] = np.conjugate(inv_A[:, :, i])
    if n_3 % 2 == 0:
        i = half_n_3
        inv_A[:, :, i] = np.linalg.inv(A[:, :, i])
        inv_A[:, :, n_3 - i] = np.conjugate(inv_A[:, :, i])
    inv_A = np.fft.ifft(inv_A)
    return inv_A


def b_circ(A):
    n_1, n_2, n_3 = A.shape
    column = np.vstack([A[:, :, i] for i in range(n_3)])
    res = [column[:, :]]
    for i in range(1, n_3):
        new_column = np.vstack([A[:, :, j % n_3] for j in range(i, i + n_3)])
        res += [new_column]
    return np.hstack(res)


def tubal_rank(A, tol=1e-10):
    n_1, n_2, n_3 = A.shape
    A = np.fft.fft(A)
    s = np.zeros((min(n_1, n_2),))
    _, diag, _ = np.linalg.svd(A[:, :, 0], full_matrices=False)
    s += diag
    half_n_3 = round(n_3 / 2)
    for i in range(1, half_n_3):
        _, diag, _ = np.linalg.svd(A[:, :, i], full_matrices=False)
        s += diag * 2
    if n_3 % 2 == 0:
        i = half_n_3
        _, diag, _ = np.linalg.svd(A[:, :, i], full_matrices=False)
        s += diag * 2
    s /= n_3
    return np.sum(s > tol)


def t_qr(A, mode="reduced"):
    n_1, n_2, n_3 = A.shape
    half_n_3 = round(n_3 / 2)
    A = np.fft.fft(A)
    if mode == "reduced" and n_1 > n_2:
        Q = np.zeros((n_1, n_2, n_3), dtype=complex)
        R = np.zeros((n_2, n_2, n_3), dtype=complex)
        Q[:, :, 0], R[:, :, 0] = np.linalg.qr(A[:, :, 0])
        for i in range(1, half_n_3):
            Q[:, :, i], R[:, :, i] = np.linalg.qr(A[:, :, i])
            Q[:, :, n_3 - i] = np.conjugate(Q[:, :, i])
            R[:, :, n_3 - i] = np.conjugate(R[:, :, i])
        if n_3 % 2 == 0:
            Q[:, :, half_n_3], R[:, :, half_n_3] = np.linalg.qr(A[:, :, half_n_3])
    else:
        Q = np.zeros((n_1, n_1, n_3), dtype=complex)
        R = np.zeros((n_1, n_2, n_3), dtype=complex)
        Q[:, :, 0], R[:, :, 0] = np.linalg.qr(A[:, :, 0], mode='complete')
        for i in range(1, half_n_3):
            Q[:, :, i], R[:, :, i] = np.linalg.qr(A[:, :, i], mode='complete')
            Q[:, :, n_3 - i] = np.conjugate(Q[:, :, i])
            R[:, :, n_3 - i] = np.conjugate(R[:, :, i])
        if n_3 % 2 == 0:
            Q[:, :, half_n_3], R[:, :, half_n_3] = np.linalg.qr(A[:, :, half_n_3], mode='complete')
    return np.fft.ifft(Q).real, np.fft.ifft(R).real
