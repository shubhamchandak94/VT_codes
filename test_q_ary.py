import numpy as np
import vt

# compute k for given n
for q in [3, 4, 8, 10, 100]:
    print('q=',q)
    for n in range(5,30):
        print('n=',n)
        print('k=',vt.find_k(n, q = q))

# compute minimum n for given k
for q in [3, 4, 8, 10, 100]:
    for k in range(1,100):
        print('k=',k)
        print('n=',vt.find_smallest_n(k, q = q))

numtrials = 10 # number of messages to try
numtrials_a_b = 5 # number of a's and b's to try
# test insertion and deletion correction for selected values of n
for q in [3, 4, 8, 10]:
    print('q',q)
    for n in [6, 7, 8, 9, 25, 50, 100]:
        print('n',n)
        k = vt.find_k(n, q = q)
        if k == 0:
            continue
        for a in np.append(0,np.random.randint(n,size=numtrials_a_b)):
            print('a',a)
            for b in np.append(0,np.random.randint(q,size=numtrials_a_b)):
                print('b',b)
                code = vt.VTCode(n, q = q, a = a)
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
                    # try insertion of 0 to q at each position
                    for i in range(n+1):
                        for ins in range(q):
                            noisy_codeword = np.insert(codeword, i, ins)
                            assert noisy_codeword.size == n+1
                            decoded_msg = code.decode(noisy_codeword)
                            assert np.array_equal(decoded_msg, msg)
