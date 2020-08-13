# Varshamov-Tenengolts (VT) codes
VT codes are error correcting codes that can correct single insertion/deletion errors with asymptotically optimal redundancy. This repository contains Python implementation of the VT code efficient encoding/decoding for both binary and general q-ary alphabets. Codes for correcting a single insertion, deletion or substitution are also supported in the binary case. The implementations are based on the following great works:

1. Varshamov, R. R., & Tenenholtz, G. M. (1965). A code for correcting a single asymmetric error. Automatica i Telemekhanika, 26(2), 288-292. [[link]](http://www.mathnet.ru/php/archive.phtml?wshow=paper&jrnid=at&paperid=11293&option_lang=eng)
2. Levenshtein, V. I. (1966, February). Binary codes capable of correcting deletions, insertions, and reversals. In Soviet physics doklady (Vol. 10, No. 8, pp. 707-710). [[link]](https://nymity.ch/sybilhunting/pdf/Levenshtein1966a.pdf)
3. Tenengolts, G. (1984). Nonbinary codes, correcting single deletion or insertion (Corresp.). IEEE Transactions on Information Theory, 30(5), 7 [[link]](https://ieeexplore.ieee.org/abstract/document/1056962/)
4. Abdel-Ghaffar, K. A., & Ferreira, H. C. (1998). Systematic encoding of the Varshamov-Tenengolts codes and the Constantin-Rao codes. IEEE Transactions on Information Theory, 44(1), 340-345. [[link]](https://ieeexplore.ieee.org/document/651063)
4. Sloane, N. J. (2000). On single-deletion-correcting codes. Codes and designs, 10, 273-291. [[link]](https://arxiv.org/abs/math/0207197)
5. Abroshan, M., Venkataramanan, R., & Fabregas, A. G. I. (2018, June). Efficient systematic encoding of non-binary VT codes. In 2018 IEEE International Symposium on Information Theory (ISIT) (pp. 91-95). IEEE. [[link]](https://ieeexplore.ieee.org/document/8437715)

Arxiv tutorial associated with this implementation: https://arxiv.org/abs/1906.07887

Also see the repository [dtch1997/single-edit-correcting-code](https://github.com/dtch1997/single-edit-correcting-code) for a quaternary alphabet specific code correcting single edits.

## Instructions

Download:
```
git clone https://github.com/shubhamchandak94/VT_codes/
```

The entire code is written in Python3 in the file `vt.py` which can be copied to another directory where you can import it using
```python
import vt
```

To initialize a code:
```
code = vt.VTCode(n, q, a, b, correct_substitutions)
```
where
```
n:                      length of codeword
q:                      alphabet size (2 for binary), q >= 2
a:                      VT code parameter (doesn't impact rate or error correction ability).
                        0 <= a < m where m =  n+1  when q = 2 and correct_substitutions = False
                                              2n+1 when q = 2 and correct_substitutions = True
                                              n    when q > 2
                        Default: a = 0
b:                      VT code parameter (doesn't impact rate or error correction ability).
                        Only relevant for q > 2.
                        0 <= b < q
                        Default: b = 0
correct_substitutions:  Only for q = 2, use one additional parity bit for ability to correct 
                        single substitution error as well. (Boolean - True/False)
                        Default: correct_substitutions = False
```

To find the message length in bits, either use `code.k` or use the function 
```python
vt.find_k(n, q, correct_substitutions)
```
To find the smallest `n` for a given message length `k`, use
```python
vt.find_smallest_n(k, q, correct_substitutions)
```

Encoding a binary message `msg` (list or numpy array of length `k`):
```python
codeword = code.encode(msg)
```
This returns a numpy array containing the encoded q-ary codeword.

Decoding a q-ary noisy codeword `noisy_codeword` (list or numpy array):
```python
decoded_msg = code.decode(noisy_codeword)
```
This returns a numpy array containing the decoded binary codeword (or None if decoding fails).

Tests to ensure correctness of the implementation are available in `test_binary.py` and `test_q_ary.py`.
