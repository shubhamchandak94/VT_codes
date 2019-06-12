import numpy as np

class VTCode:
    def __init__(self, n: int, q: int, a = 0: int, b = 0: int,
                correct_substitutions = False):
        '''
        Here n is the codeword length and q is the alphabet size.
        a and b are parameters of the code that do not impact the rate in this
        implementation (so can be left at their default values).
        Set correct_substitutions to True for q = 2 if you want ability to correct
        single substitution errors as well.
        '''
        assert q >= 2
        assert n >= 2
        self.n = n
        self.q = q
        self.correct_substitutions = correct_substitutions
        self.k = find_k(self.n, self.q, self.correct_substitutions)
        assert self.k > 0
        self.a = a
        self.b = b
        if self.q == 2:
            if not self.correct_substitutions:
                self.m = self.n + 1
            else:
                self.m = 2*self.n + 1
            assert 0 <= self.a < self.m
            self._generate_systematic_positions_binary()
        else:
            self.m = self.n
            assert 0 <= self.a < self.m
            assert self.b < self.q

    def decode(self, y):
        '''
        input  y: list or 1d np array with the noisy codeword
        return x: decoded message bits as a 1d numpy array with dtype uint32 or
                  None if decoding fails
        '''
        y = np.array(y, dtype=np.uint32)
        assert y.ndim == 1
        n_y = y.size
        if (n_y < self.n - 1) or (n_y > self.n + 1):
            return None
        if (np.max(y) > self.q-1) or (np.min(y) < 0):
            print("Value in y out of range 0...q-1")
            raise RuntimeError
        if self.q == 2:
            if n_y != self.n:
                y = _correct_binary_indel(self.n, self.m, self.a, y)
            else:
                if self.correct_substitutions and not self._is_codeword(y):
                    y = _correct_binary_substitution(n, m, a, y)
        else:
            if n_y != self.n:
                y = _correct_q_ary_indel(self.n, self.m, self.a, self.b, self.q, y)
        return self._decode_codeword(y)

    def encode(self, x):
        '''
        input  x: list or 1d np array with the message bits (length k)
        return y: encoded codeword as a 1d numpy array with dtype uint32 (length n)
        '''
        x = np.array(x, dtype = np.uint32)
        assert x.ndim == 1
        assert x.size == self.k
        if (np.max(x) > 1) or (np.min(x) < 0):
            print("Value in x out of range {0, 1}")
            raise RuntimeError
        if self.q == 2:
            return _encode_binary(self,x)
        else:
            return _encode_q_ary(self,x)


    def _decode_codeword(self, y):
        '''
        decode a codeword (if not a codeword, returns None)
        '''
        if not _is_codeword(y):
            return None
        if self.q == 2:
            return _decode_codeword_binary(self,y)
        else:
            return _decode_codeword_q_ary(self,y)

    def _decode_codeword_binary(self, y):
        '''
        decoding helper for binary case (assume it's a valid codeword)
        '''
        # just return values at the systematic positions
        return y[self.systematic_positions]

    def _decode_codeword_q_ary(self, y):
        '''
        decoding helper for q ary case
        '''
        raise NotImplementedError

    def _encode_binary(self, x):
        '''
        encoding helper for binary case
        '''
        y = np.zeros(self.n, dtype = np.uint32)
        # first set systematic positions
        y[self.systematic_positions-1] = x
        # now set the rest positions based on syndrome
        syndrome = _compute_syndrome_binary(self.n, self.m, self.a, y)
        if syndrome != 0:
            for pos in reversed(self.parity_positions):
                if syndrome >= pos:
                    y[pos-1] = 1
                    syndrome -= pos
                    if syndrome == 0:
                        break
        assert self._is_codeword(y)
        return y

    def _encode_q_ary(self, x):
        '''
        encoding helper for q ary case
        '''
        raise NotImplementedError

    def _is_codeword(self, y):
        '''
        return True if y is a codeword
        '''
        if y == None or y.size != self.n:
            return False
        if self.q == 2:
            return (_compute_syndrome_binary(self.n, self.m, self.a, y) == 0)
        else:
            return (_compute_syndrome_q_ary(self.n, self.m, self.a, self.b, self.q, y) == (0,0))

    def _generate_systematic_positions_binary(self):
        # generate positions of systematic and parity bits (1 indexed)
        t = np.ceil(np.log(self.n+1)/np.log(2)).astype(np.uint32)
        # put powers of two in the parity positions
        self.parity_positions = np.zeros(self.n-self.k, dtype=np.uint32)
        for i in range(t):
            self.parity_positions[i] = np.pow(2,i)
        if self.correct_substitutions:
            assert self.parity_positions.size == t + 1
            # one extra parity bit in this case
            # depending on if last position in codeword is already filled,
            # set it or the previous position as a parity_position
            if self.parity_positions[t-1] == self.n:
                self.parity_positions[t-1] = self.n - 1
                self.parity_positions[t] = self.n
            else:
                self.parity_positions[t] = self.n
        self.systematic_positions =  np.setdiff1d(np.arange(1,self.n+1), self.parity_positions)
        return

# utility functions

def find_smallest_n(k: int, q : int, correct_substitutions = False):
    '''
    Returns smallest n for a code with given k and q.
    Here k is the message length in bits, n is the codeword length and q is the
    alphabet size.
    Set correct_substitutions to True for q = 2 if you want ability to correct
    single substitution errors as well.
    '''
    assert q >= 2
    assert k >= 1
    if q != 2 and correct_substitutions == True:
        print("correct_substitutions can be True only for q = 2")
        raise RuntimeError
    # set the starting n and then increase till you get the minimum
    if q == 2
        if not correct_substitutions:
            n = k + np.ceil(np.log(k+1)/np.log(2)).astype(np.uint32)
        else:
            n = k + np.ceil(np.log(2*k+1)/np.log(2)).astype(np.uint32)
    else:
        raise NotImplementedError
    while True:
        if find_k(n, q, correct_substitutions) >= k:
            break
        n += 1
    return n

def find_k(n: int, q: int, correct_substitutions = False):
    '''
    Returns k for a code with given n and q.
    Here k is the message length in bits, n is the codeword length and q is the
    alphabet size.
    Set correct_substitutions to True for q = 2 if you want ability to correct
    single substitution errors as well.
    '''
    if q != 2 and correct_substitutions:
        print("correct_substitutions can be True only for q = 2")
        raise RuntimeError
    if q == 2
        if not correct_substitutions:
            return n - np.ceil(np.log(n+1)/np.log(2)).astype(np.uint32)
        else:
            return n - np.ceil(np.log(2*n+1)/np.log(2)).astype(np.uint32)
    else:
        raise NotImplementedError


# internal functions
def _correct_binary_indel(n: int, m: int, a: int, y):
    '''
    correct single insertion deletion error in the binary case.
    Used also in the q-ary alphabet case as a subroutine.
    Input: n (codeword length), m (modulus), a (syndrome),
           y (noisy codeword, np array)
    Output: corrected codeword
    '''
    raise NotImplementedError

def _correct_binary_substitution(n: int, m: int, a: int, y):
    '''
    correct single substitution error in the binary case.
    Input: n (codeword length), m (modulus), a (syndrome),
           y (noisy codeword, np array)
    Output: corrected codeword
    '''
    raise NotImplementedError

def _compute_syndrome_binary(n: int, m: int, a: int, y):
    '''
    compute the syndrome in the binary case (a - sum(i*y_i) mod m)
    '''
    n_y = y.size
    return np.mod(a - (1+np.arange(n_y))*y,m)

def _correct_q_ary_indel(n: int, m: int, a: int, b: int, q: int, y):
    '''
    correct single insertion/deletion error in the q-ary case.
    Input: n (codeword length), m (modulus), a, b (syndrome), q (alphabet)
           y (noisy codeword, np array)
    Output: corrected codeword
    '''
    raise NotImplementedError

def _compute_syndrome_q_ary(n: int, m: int, a: int, b: int, q: int, y):
    '''
    compute the syndrome in the binary case (a - sum(i*alpha_i) mod m, b - sum(y_i) mod q)
    '''
    n_y = y.size
    alpha = _convert_y_to_alpha(y)
    return (np.mod(a - (1+np.arange(n_y-1))*alpha,m), np.mod(b-np.sum(y),q))

def _convert_y_to_alpha(y):
    '''
    convert q-ary y of length n to binary length n-1 alpha, alpha_i = 1 iff y_i >= y_{i-1}
    '''
    return (y[1:] >= y[:-1]).astype(np.uint32)
