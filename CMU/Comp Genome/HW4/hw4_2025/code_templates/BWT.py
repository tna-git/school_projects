# BWT.py
# HW1, Computational Genomics, Spring 2024
# andrewid: tnair

# These code signatures are for your benefit. This problem is not being graded by autograder, but you must still turn it in.

def rle(s):
    """Run Length Encoder
    Args: s, string to be encoded
    Returns: RLE(s)
    """
    rle = ""
    count = 1
    for i in range(1, len(s)):
        if s[i] != s[i-1]:
            if count > 1:
                rle += s[i-1] + s[i-1] + str(count)
            else:
                rle += s[i-1]
            count = 1
        else:
            count += 1
    
    # last run
    if count > 1:
        rle += s[-1] + s[-1] + str(count)
    else:
        rle += s[-1]
    return rle

def bwt_encode(s):
    """Burrows-Wheeler Transform
    Args: s, string, which must not contain '{' or '}'
    Returns: BWT(s), which contains '{' and '}'
    """
    s = '{' + s + '}'
    rotations = [s[i:] + s[:i] for i in range(len(s))]
    rotations.sort()
    bwt = ''.join(rotation[-1] for rotation in rotations)
    return bwt


def bwt_decode(bwt):
    """Inverse Burrows-Wheeler Transform
    Args: bwt, BWT'ed string, which should contain '{' and '}'
    Returns: reconstructed original string s, must not contains '{' or '}'
    """
    n = len(bwt)
    table = [''] * n
    for _ in range(n):
        table = sorted(bwt[i] + table[i] for i in range(n))
    for row in table:
        if row.startswith('{') and row.endswith('}'):
            return row[1:-1]

def test_string(s):
    compressed = rle(s)
    bwt = bwt_encode(s)
    compressed_bwt = rle(bwt)
    reconstructed = bwt_decode(bwt)
    template = "{:25} ({:3d}) {}"
    print(template.format("original", len(s), s))
    print(template.format("bwt_enc(orig)", len(bwt), bwt))
    print(template.format("bwt_dec(bwt_enc(orig))", len(reconstructed), reconstructed))
    print(template.format("rle(orig)", len(compressed), compressed))
    print(template.format("rle(bwt_enc(orig))", len(compressed_bwt), compressed_bwt))
    print("2b length:", len(rle("mississippiicecreammississippi")))
    print("2g length:", len(rle(bwt_encode("mississippiicecreammississippi"))))
    print(len("abcdefghijklmnopqristuvwxyz1234567890[]\';.,!@#$%^&*()~`abcdefghijklmnopqristuvwxyz1234567890[]\';.,!@#$%^&*()~`"))
    print("2i.i length:", len(rle(bwt_encode("abcdefghijklmnopqristuvwxyz1234567890[]\';.,!@#$%^&*()~`abcdefghijklmnopqristuvwxyz1234567890[]\';.,!@#$%^&*()~`"))))
    print(len("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))
    print("2i.ii length:", len(rle(bwt_encode("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))))
if __name__ == "__main__":
    # Add more of your own strings to explore for question (i)
    test_strings = ["WOOOOOHOOOOHOOOO!",
                    "scottytartanscottytartanscottytartanscottytartan"]
    for s in test_strings:
        test_string(s)
