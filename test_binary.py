import numpy as np
import vt

# compute k for given n
for n in range(100):
    print('n=',n)
    print('k=',vt.find_k(n, q = 2),'for no substitution correction')
    print('k=',vt.find_k(n, q = 2, correct_substitutions=True),'for substitution correction')

# compute minimum n for given k
for k in range(1,100):
    print('k=',k)
    print('n=',vt.find_smallest_n(k, q = 2),'for no substitution correction')
    print('n=',vt.find_smallest_n(k, q = 2, correct_substitutions=True),'for substitution correction')

numtrials = 10 # number of messages to try
numtrials_a = 10 # number of a's to try
# test insertion and deletion correction for selected values of n
for n in [5, 10, 15, 50, 100]:
    for a in np.append(0,np.random.randint(n+1,size=numtrials_a)):
        code = vt.VTCode(n, q = 2, a = a)
        k = vt.find_k(n, q = 2)
        for trial in range(numtrials):
            msg = np.random.choice([0,1], k)
            codeword = code.encode(msg)
            assert codeword.size == n
            # first test decoding for given codeword
            decoded_msg = code.decode(codeword)
            assert np.array_equal(decoded_msg, msg)
            # try deletion at each position
            for i in range(n):
                noisy_codeword = np.delete(codeword, i)
                assert noisy_codeword.size == n-1
                decoded_msg = code.decode(noisy_codeword)
                assert np.array_equal(decoded_msg, msg)
            # try insertion of 0 and 1 at each position
            for i in range(n+1):
                for bit in range(2):
                    noisy_codeword = np.insert(codeword, i, bit)
                    assert noisy_codeword.size == n+1
                    decoded_msg = code.decode(noisy_codeword)
                    assert np.array_equal(decoded_msg, msg)

# test insertion, deletion and substitution correction when correct_substitutions is True
for n in [5, 10, 15, 50, 100]:
    for a in np.append(0,np.random.randint(n+1,size=numtrials_a)):
        code = vt.VTCode(n, q = 2, a = a,correct_substitutions = True)
        k = vt.find_k(n, q = 2, correct_substitutions = True)
        for trial in range(numtrials):
            msg = np.random.choice([0,1], k)
            codeword = code.encode(msg)
            assert codeword.size == n
            # first test decoding for given codeword
            decoded_msg = code.decode(codeword)
            assert np.array_equal(decoded_msg, msg)
            # try deletion at each position
            for i in range(n):
                noisy_codeword = np.delete(codeword, i)
                assert noisy_codeword.size == n-1
                decoded_msg = code.decode(noisy_codeword)
                assert np.array_equal(decoded_msg, msg)
            # try insertion of 0 and 1 at each position
            for i in range(n+1):
                for bit in range(2):
                    noisy_codeword = np.insert(codeword, i, bit)
                    assert noisy_codeword.size == n+1
                    decoded_msg = code.decode(noisy_codeword)
                    assert np.array_equal(decoded_msg, msg)
            # try substitution at each position
            for i in range(n):
                noisy_codeword = np.array(codeword)
                noisy_codeword[i] = 1 - noisy_codeword[i]
                assert noisy_codeword.size == n
                decoded_msg = code.decode(noisy_codeword)
                assert np.array_equal(decoded_msg, msg)
