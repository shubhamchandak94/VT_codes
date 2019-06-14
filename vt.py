import numpy as np

class VTCode:
    def __init__(self, n: int, q: int, a = 0, b = 0,
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
            self.t = np.ceil(np.log2(n)).astype(np.int64)
            assert 0 <= self.a < self.m
            assert 0 <= self.b < self.q
            self._generate_tables()

    def decode(self, y):
        '''
        input  y: list or 1d np array with the noisy codeword
        return x: decoded message bits as a 1d numpy array with dtype int64 or
                  None if decoding fails
        '''
        y = np.array(y, dtype=np.int64)
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
                    y = _correct_binary_substitution(self.n, self.m, self.a, y)
        else:
            if n_y != self.n:
                y = _correct_q_ary_indel(self.n, self.m, self.a, self.b, self.q, y)
        return self._decode_codeword(y)

    def encode(self, x):
        '''
        input  x: list or 1d np array with the message bits (length k)
        return y: encoded codeword as a 1d numpy array with dtype int64 (length n)
        '''
        x = np.array(x, dtype = np.int64)
        assert x.ndim == 1
        assert x.size == self.k
        if (np.max(x) > 1) or (np.min(x) < 0):
            print("Value in x out of range {0, 1}")
            raise RuntimeError
        if self.q == 2:
            return self._encode_binary(x)
        else:
            return self._encode_q_ary(x)


    def _decode_codeword(self, y):
        '''
        decode a codeword (if not a codeword, returns None)
        '''
        if not self._is_codeword(y):
            return None
        if self.q == 2:
            return self._decode_codeword_binary(y)
        else:
            return self._decode_codeword_q_ary(y)

    def _decode_codeword_binary(self, y):
        '''
        decoding helper for binary case (assume it's a valid codeword)
        '''
        # just return values at the systematic positions
        return y[self.systematic_positions-1]

    def _decode_codeword_q_ary(self, y):
        '''
        decoding helper for q ary case
        '''
        x = np.zeros(self.k, dtype = np.int64)
        # step 1
        step_1_num_bits = np.floor(np.max([self.n-3*self.t+3,0])*np.log2(self.q)).astype(np.int64)
        if step_1_num_bits > 0:
            step_1_bits = _convert_base(y[self.systematic_positions_step_1], self.q, 2, step_1_num_bits)
            if step_1_bits is None: # if more than expected bits
                return None
            x[:step_1_num_bits] = step_1_bits

        # step 2
        bits_done = step_1_num_bits
        bits_per_tuple_step_2 = np.floor(2*np.log2(self.q-1)).astype(np.int64)
        for j in range(3, self.t):
            if 2**j == self.n-1:
                # special case, here we store np.floor(np.log2(q-1)) bits in 2**j - 1
                num_bits_special_case = np.floor(np.log2(self.q-1)).astype(np.int64)
                # y[2**j-1] can be from 1 to q-1
                if y[2**j-1] == 0:
                    return None
                x[bits_done:bits_done+num_bits_special_case] = \
                    _number_to_q_ary_array(y[2**j-1]-1, 2, num_bits_special_case)
                bits_done += num_bits_special_case
                break
            if (y[2**j-1], y[2**j+1]) in self.table_1_rev:
                x[bits_done:bits_done+bits_per_tuple_step_2] = \
                    _number_to_q_ary_array(self.table_1_rev[(y[2**j-1], y[2**j+1])], 2, bits_per_tuple_step_2)
            else:
                return None
            bits_done += bits_per_tuple_step_2

        if self.q == 3:
            if y[5] != 2:
                return None
        else:
            if y[3] != self.q - 1:
                return None
            bits_in_c_5 = np.floor(np.log2(self.q-1)).astype(np.int64)
            if y[5] not in self.table_2_rev:
                return None
            else:
                x[bits_done:bits_done+bits_in_c_5] = \
                    _number_to_q_ary_array(self.table_2_rev[y[5]], 2, bits_in_c_5)
            bits_done += bits_in_c_5
        assert bits_done == self.k
        return x

    def _encode_binary(self, x):
        '''
        encoding helper for binary case
        '''
        y = np.zeros(self.n, dtype = np.int64)
        # first set systematic positions
        y[self.systematic_positions-1] = x
        # now set the rest positions based on syndrome
        syndrome = _compute_syndrome_binary(self.m, self.a, y)
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
        y = np.zeros(self.n, dtype = np.int64)
        # step 1 (encode bits in positions not near dyadic)
        step_1_num_bits = np.floor(np.max([self.n-3*self.t+3,0])*np.log2(self.q)).astype(np.int64)
        if step_1_num_bits > 0:
            y[self.systematic_positions_step_1] = _convert_base(x[:step_1_num_bits],\
                    2, self.q, out_size = self.systematic_positions_step_1.size)
        # step 2 (encode bits in positions near dyadic, but not at dyadic)
        bits_done = step_1_num_bits
        bits_per_tuple_step_2 = np.floor(2*np.log2(self.q-1)).astype(np.int64)
        for j in range(3, self.t):
            table_1_index = _q_ary_array_to_number(x[bits_done:bits_done+bits_per_tuple_step_2], 2)
            if 2**j == self.n-1:
                # special case, here we store np.floor(np.log2(q-1)) bits in 2**j - 1
                num_bits_special_case = np.floor(np.log2(self.q-1)).astype(np.int64)
                y[2**j-1] = _q_ary_array_to_number(x[bits_done:bits_done+num_bits_special_case], 2)+1
                # so y[2**j-1] can be from 1 to q-1
                bits_done += num_bits_special_case
                break
            y[2**j-1] = self.table_1_r[table_1_index]
            y[2**j+1] = self.table_1_l[table_1_index]
            bits_done += bits_per_tuple_step_2
        y[3] = self.q - 1
        if self.q == 3:
            y[5] = 2
        else:
            bits_in_c_5 = np.floor(np.log2(self.q-1)).astype(np.int64)
            table_2_index = _q_ary_array_to_number(x[bits_done:bits_done+bits_in_c_5],2)
            y[5] = self.table_2[table_2_index]
            bits_done += bits_in_c_5
        assert bits_done == self.k
        # step 3 (set alpha at positions except dyadic)
        alpha = _convert_y_to_alpha(y)
        for j in range(2, self.t):
            if 2**j == self.n-1:
                break
            alpha[2**j+1-1] = (y[2**j+1] >= y[2**j-1])
        alpha[3-1] = 1
        # step 4 (set alpha at dyadic positions using VT conditions)
        # first set alpha at dyadic positions to 0
        for j in range(self.t):
            alpha[2**j-1] = 0
        syndrome = _compute_syndrome_binary(self.m, self.a, alpha)
        if syndrome != 0:
            for j in reversed(range(self.t)):
                pos = 2**j
                if syndrome >= pos:
                    alpha[pos-1] = 1
                    syndrome -= pos
                    if syndrome == 0:
                        break
        # step 5 (set symbols of y at dyadic positions except 1 and 2)
        for j in range(2,self.t):
            pos = 2**j
            if alpha[pos-1] == 0:
                y[pos] = y[pos-1]-1
            else:
                y[pos] = y[pos-1]
        # step 6 (set positions 0, 1 and 2)
        w = np.mod(self.b-np.sum(y[3:]),self.q)
        if self.q == 3:
            if alpha[1-1] == 1 and alpha[2-1] == 1:
                y[2], y[1], y[0] = 2, 2, np.mod(w-4, 3)
            elif alpha[1-1] == 1 and alpha[2-1] == 0:
                y[2], y[1], y[0] = 1, 2, w
            elif alpha[1-1] == 0 and alpha[2-1] == 1:
                y[2] = 2
                if w == 1:
                    y[1], y[0] = 0, 2
                elif w == 0:
                    y[1], y[0] = 0, 1
                else:
                    y[1], y[0] = 1, 2
            else:
                alpha[1-1], alpha[2-1], alpha[3-1] = 1, 1, 0
                y[3] = 1
                y[4] = 0 if (alpha[4-1] == 0) else 1
                y[2], y[1] = 2, 2
                y[0] = np.mod(self.b-np.sum(y[1:]), 3)
        else:
            # get 0 <= x_ < y_ < z_ <= q-1 such that x_+y_+z_ = w mod q
            if w == 1:
                x_, y_, z_ = 0, 2, self.q - 1
            elif w == 2:
                x_, y_, z_ = 1, 2, self.q - 1
            else:
                x_, y_, z_ = 0, 1, np.mod(w-1, self.q)
            # now we assign x_, y_, z_ to  y[0], y[1], y[2] to satisfy the alpha conditions
            if alpha[1-1] == 0 and alpha[2-1] == 0:
                y[0], y[1], y[2] = z_, y_, x_
            elif  alpha[1-1] == 0 and alpha[2-1] == 1:
                y[0], y[1], y[2] = z_, x_, y_
            elif  alpha[1-1] == 1 and alpha[2-1] == 0:
                y[0], y[1], y[2] = x_, z_, y_
            else:
                y[0], y[1], y[2] = x_, y_, z_
        assert np.array_equal(alpha, _convert_y_to_alpha(y))
        assert self._is_codeword(y)
        return y

    def _is_codeword(self, y):
        '''
        return True if y is a codeword
        '''
        if y is None or y.size != self.n:
            return False
        if self.q == 2:
            return (_compute_syndrome_binary(self.m, self.a, y) == 0)
        else:
            return (_compute_syndrome_q_ary(self.m, self.a, self.b, self.q, y) == (0,0))

    def _generate_systematic_positions_binary(self):
        # generate positions of systematic and parity bits (1 indexed)
        t = np.ceil(np.log2(self.n+1)).astype(np.int64)
        # put powers of two in the parity positions
        self.parity_positions = np.zeros(self.n-self.k, dtype=np.int64)
        for i in range(t):
            self.parity_positions[i] = np.power(2,i)
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

    def _generate_tables(self):
        '''
        generate relevant tables for encoding in q-ary case
        '''
        # table 1: map floor(2*log2(q-1)) bits to pairs (r,l) of q-ary symbols
        # such that r != 0 and l != r-1
        table_1_size = 2**(np.floor(2*np.log2(self.q-1)).astype(np.int64))
        self.table_1_l = np.zeros(table_1_size, dtype = np.int64)
        self.table_1_r = np.zeros(table_1_size, dtype = np.int64)
        pos_in_table = 0
        for r in range(self.q):
            if pos_in_table == table_1_size:
                break
            if r == 0:
                continue
            for l in range(self.q):
                if pos_in_table == table_1_size:
                    break
                if l == r-1:
                    continue
                self.table_1_l[pos_in_table] = l
                self.table_1_r[pos_in_table] = r
                pos_in_table += 1

        # reverse table for decoding
        self.table_1_rev = {(self.table_1_r[i],self.table_1_l[i]): i for i in range(table_1_size)}

        # table 2: map floor(log2(q-1)) bits to 1 q-ary symbol != q-2
        if self.q != 3: # 3 is special case
            table_2_size = 2**(np.floor(np.log2(self.q-1)).astype(np.int64))
            self.table_2 = np.zeros(table_2_size, dtype = np.int64)
            for i in range(table_2_size):
                self.table_2[i] = i
                if i == self.q-2:
                    self.table_2[i] = self.q-1
            self.table_2_rev = {self.table_2[i]: i for i in range(table_2_size)}

        # also generate systematic positions for step 1 of encoding (0-indexed)
        # all positions except for 0, 1, 2, 3, 4, 5, 2^j-1, 2^j, 2^j+1
        non_systematic_pos = [0, 1, 2, 3, 4, 5]
        for j in range(3,self.t):
            non_systematic_pos = non_systematic_pos + [2**j-1, 2**j, 2**j+1]
        non_systematic_pos = np.array(non_systematic_pos, dtype = np.int64)
        self.systematic_positions_step_1 =  np.setdiff1d(np.arange(self.n), non_systematic_pos)
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
    if q == 2:
        if not correct_substitutions:
            n = k + np.ceil(np.log2(k+1)).astype(np.int64)
        else:
            n = k + np.ceil(np.log2(2*k+1)).astype(np.int64)
    else:
        n = int(k/np.ceil(np.log2(q)).astype(np.int64))
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
    if q == 2:
        if not correct_substitutions:
            return n - np.ceil(np.log2(n+1)).astype(np.int64)
        else:
            return n - np.ceil(np.log2(2*n+1)).astype(np.int64)
    else:
        t = np.ceil(np.log2(n)).astype(np.int64)
        if q == 3:
            if n < 7:
                return 0
            if _power_of_two(n-1):
                # in this case we can't store data in 2**(t-1)+1
                return np.floor((n-3*t+3)*np.log2(q)).astype(np.int64) + 2*(t-4) + 1
            else:
                return np.floor((n-3*t+3)*np.log2(q)).astype(np.int64) + 2*(t-3)
        else:
            if n < 6:
                return 0
            if _power_of_two(n-1):
                # in this case we can't store data in 2**(t-1)+1
                return np.floor(np.max([(n-3*t+3),0])*np.log2(q)).astype(np.int64) + \
                    np.floor(2*np.log2(q-1)).astype(np.int64)*np.max([(t-4),0]) + \
                    2*np.floor(np.log2(q-1)).astype(np.int64)
            else:
                return np.floor(np.max([(n-3*t+3),0])*np.log2(q)).astype(np.int64) + \
                    np.floor(2*np.log2(q-1)).astype(np.int64)*np.max([(t-3),0]) + \
                    np.floor(np.log2(q-1)).astype(np.int64)


# internal functions
def _correct_binary_indel(n: int, m: int, a: int, y):
    '''
    correct single insertion deletion error in the binary case.
    Used also in the q-ary alphabet case as a subroutine.
    Input: n (codeword length), m (modulus), a (syndrome),
           y (noisy codeword, np array)
    Output: corrected codeword
    '''
    s = _compute_syndrome_binary(m, a, y)
    w = np.sum(y)
    y_decoded = np.zeros(n, dtype=np.int64)
    if y.size == n-1:
        # deletion
        if s == 0:
            # last entry 0 was deleted
            y_decoded[:-1] = y
        elif s <= w:
            # 0 deleted and s = number of 1s to right
            num_ones_seen = 0
            for i in reversed(range(n-1)):
                if y[i] == 1:
                    num_ones_seen += 1
                    if num_ones_seen == s:
                        y_decoded[:i] = y[:i]
                        y_decoded[i+1:] = y[i:]
                        break
        else:
            # 1 deleted and s-w-1 = number of 0s to left
            num_zeros_seen = 0
            if s-w-1 == 0:
                y_decoded[0] = 1
                y_decoded[1:] = y
            else:
                success = False
                for i in range(n-1):
                    if y[i] == 0:
                        num_zeros_seen += 1
                        if num_zeros_seen == s-w-1:
                            y_decoded[:i+1] = y[:i+1]
                            y_decoded[i+1] = 1
                            y_decoded[i+2:] = y[i+1:]
                            success = True
                            break
                if not success:
                    y_decoded = None
    else:
        # insertion
        if s == m-n-1 or s == 0:
            # last entry inserted
            y_decoded = y[:-1]
        elif s == m-w:
            # remove first entry
            y_decoded = y[1:]
        elif s > m-w:
            # 0 was inserted, m-s 1's to the right of this zero
            num_ones_seen = 0
            success = False
            for i in reversed(range(2,n+1)):
                if y[i] == 1:
                    num_ones_seen += 1
                    if num_ones_seen == m-s:
                        if y[i-1] == 0:
                            y_decoded[:i-1] = y[:i-1]
                            y_decoded[i-1:] = y[i:]
                            success = True
                        else:
                            pass
                        break
            if not success:
                y_decoded = None
        else:
            # 1 was inserted, m-w-s 0's to the left of this 1
            num_zeros_seen = 0
            success = False
            for i in range(n-1):
                if y[i] == 0:
                    num_zeros_seen += 1
                    if num_zeros_seen == m-w-s:
                        if y[i+1] == 1:
                            y_decoded[:i+1] = y[:i+1]
                            y_decoded[i+1:] = y[i+2:]
                            success = True
                        else:
                            pass
                        break
            if not success:
                y_decoded = None
    return y_decoded

def _correct_binary_substitution(n: int, m: int, a: int, y):
    '''
    correct single substitution error in the binary case.
    Input: n (codeword length), m (modulus), a (syndrome),
           y (noisy codeword, np array)
    Output: corrected codeword
    '''
    assert m == 2*n+1
    s = _compute_syndrome_binary(m, a, y)
    y_decoded = np.array(y)
    if s == 0:
        # no error, nothing to do
        pass
    elif s < n+1:
        # 1 flipped to 0 at s
        y_decoded[s-1] = 1
    else:
        # 0 flipped to 1 at 2n+1-s
        y_decoded[2*n+1-s-1] = 0
    return y_decoded

def _compute_syndrome_binary(m: int, a: int, y):
    '''
    compute the syndrome in the binary case (a - sum(i*y_i) mod m)
    '''
    n_y = y.size
    return np.mod(a - np.sum((1+np.arange(n_y))*y),m)

def _correct_q_ary_indel(n: int, m: int, a: int, b: int, q: int, y):
    '''
    correct single insertion/deletion error in the q-ary case.
    Input: n (codeword length), m (modulus), a, b (syndrome), q (alphabet)
           y (noisy codeword, np array)
    Output: corrected codeword
    '''
    alpha = _convert_y_to_alpha(y)
    alpha_corrected = _correct_binary_indel(n-1, m, a, alpha)
    if alpha_corrected is None or _compute_syndrome_binary(m, a, alpha_corrected) != 0:
        return None
    y_decoded = np.zeros(n, dtype=np.int64)
    if alpha.size == n-2:
        # deletion
        error_symbol = np.mod(b-np.sum(y),q) # value of symbol deleted
        # first find the position where alpha and alpha_corrected differ
        if np.array_equal(alpha, alpha_corrected[:-1]):
            diff_pos = n-2
        else:
            for diff_pos in range(n-2):
                if alpha[diff_pos] != alpha_corrected[diff_pos]:
                    break
        # at this point we know that alpha_corrected[diff_pos] was deleted from
        # the run containing diff_pos position
        # now we move back from diff pos and try to find the position
        del_pos_found = False
        for del_pos in reversed(range(diff_pos + 2)):
            if del_pos == 0:
                if alpha_corrected[0] == (y[0] >= error_symbol):
                    del_pos_found = True
                    break
            elif del_pos == n-1:
                if (alpha_corrected[n-2] == (error_symbol >= y[n-2])):
                    del_pos_found = True
                    break
            else:
                if (alpha_corrected[del_pos-1] == (error_symbol >= y[del_pos-1])) \
                and (alpha_corrected[del_pos+1-1] == (y[del_pos] >= error_symbol)):
                    del_pos_found = True
                    break
        if del_pos_found:
            y_decoded[:del_pos] = y[:del_pos]
            y_decoded[del_pos] = error_symbol
            y_decoded[del_pos+1:] = y[del_pos:]
        else:
            y_decoded = None
    else:
        #insertion
        # first find the position where alpha and alpha_corrected differ
        error_symbol = np.mod(np.sum(y)-b,q) # value of symbol inserted
        if np.array_equal(alpha[:-1], alpha_corrected):
            diff_pos = n-1
        else:
            for diff_pos in range(n):
                if alpha[diff_pos] != alpha_corrected[diff_pos]:
                    break
        # at this point we know that alpha_corrected[diff_pos] was inserted in
        # the run containing diff_pos position
        # now we move back from diff pos and try to find the position
        ins_pos_found = False
        for ins_pos in reversed(range(diff_pos + 2)):
            if ins_pos == 0 or ins_pos == n:
                if (y[ins_pos] == error_symbol):
                    ins_pos_found = True
                    break
            else:
                if (y[ins_pos] == error_symbol) and \
                    (alpha_corrected[ins_pos-1] == (y[ins_pos+1] >= y[ins_pos-1])):
                    ins_pos_found = True
                    break
        if ins_pos_found:
            y_decoded[:ins_pos] = y[:ins_pos]
            y_decoded[ins_pos:] = y[ins_pos+1:]
        else:
            y_decoded = None
    if y_decoded is not None and _compute_syndrome_q_ary(m, a, b, q, y_decoded) == (0,0):
        return y_decoded
    else:
        return None

def _compute_syndrome_q_ary(m: int, a: int, b: int, q: int, y):
    '''
    compute the syndrome in the binary case (a - sum(i*alpha_i) mod m, b - sum(y_i) mod q)
    '''
    n_y = y.size
    alpha = _convert_y_to_alpha(y)
    return (_compute_syndrome_binary(m, a, alpha), np.mod(b-np.sum(y),q))

def _convert_y_to_alpha(y):
    '''
    convert q-ary y of length n to binary length n-1 alpha, alpha_i = 1 iff y_i >= y_{i-1}
    '''
    return (y[1:] >= y[:-1]).astype(np.int64)

def _q_ary_array_to_number(q_ary_array, q):
    # convert q_ary_array (MSB first) to a number
    num = 0
    for i in q_ary_array:
        i = i.item() # to prevent overflow, convert to python native types
        num = q*num + i
    return num

def _number_to_q_ary_array(num, q, out_size = None):
    # convert number to q_ary_array (MSB first)
    # pad to outsize
    out_array = []
    while num > 0:
        out_array.append(num%q)
        num //= q
    out_array.reverse()
    out_array = np.array(out_array, dtype = np.int64)
    if out_size == None:
        return out_array
    if out_array.size > out_size:
        return None
    else:
        return np.pad(out_array, (out_size - out_array.size,0), 'constant')

def _convert_base(in_array, in_base, out_base, out_size = None):
    # convert in_array represented in in_base to array in out_base (both MSB first)
    # pad to out_size
    return _number_to_q_ary_array(_q_ary_array_to_number(in_array, in_base), out_base, out_size)

def _power_of_two(num):
    '''
    return True if num is a power of 2
    '''
    return np.ceil(np.log2(num)).astype(np.int64) == np.floor(np.log2(num)).astype(np.int64)
