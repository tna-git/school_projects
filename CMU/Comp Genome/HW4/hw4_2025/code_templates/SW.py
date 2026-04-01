# SW.py
# HW1, Computational Genomics, Spring 2024
# andrewid: tnair

# WARNING: Do not change the file name; Autograder expects it.

import sys

def ReadFASTA(filename):
    fp=open(filename, 'r')
    Sequences={}
    tmpname=""
    tmpseq=""
    for line in fp:
        if line[0]==">":
            if len(tmpseq)!=0:
                Sequences[tmpname]=tmpseq
            tmpname=line.strip().split()[0][1:]
            tmpseq=""
        else:
            tmpseq+=line.strip()
    Sequences[tmpname]=tmpseq
    fp.close()
    return Sequences

# You may define any helper functions for Smith-Waterman algorithm here

# Do not change this function signature
def smith_waterman(seq1, seq2):
    """Find the best local alignment for seq1 and seq2
    Returns: 3 items as so:
    the alignment score, alignment in seq1 (str), alignment in seq2 (str)
    """
    match = 2
    mismatch = -1
    gap = -1

    n = len(seq1)
    m = len(seq2)

    # DP matrix for scores, and a matrix to store traceback directions
    # traceback values: 0 = stop, 1 = diag, 2 = up, 3 = left
    H = [[0] * (m + 1) for _ in range(n + 1)]
    trace = [[0] * (m + 1) for _ in range(n + 1)]

    max_score = 0
    max_pos = (0, 0)

    # Fill DP matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = H[i - 1][j - 1] + (match if seq1[i - 1] == seq2[j - 1] else mismatch)
            up = H[i - 1][j] + gap
            left = H[i][j - 1] + gap
            best = max(0, diag, up, left)

            H[i][j] = best
            if best == 0:
                trace[i][j] = 0
            elif best == diag:
                trace[i][j] = 1
            elif best == up:
                trace[i][j] = 2
            else:  # best == left
                trace[i][j] = 3

            if best > max_score:
                max_score = best
                max_pos = (i, j)

    # Traceback from max_pos until we hit a cell with score 0
    i, j = max_pos
    align1 = []
    align2 = []

    while i > 0 and j > 0 and H[i][j] != 0:
        direction = trace[i][j]
        if direction == 1:  # diag
            align1.append(seq1[i - 1])
            align2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif direction == 2:  # up (gap in seq2)
            align1.append(seq1[i - 1])
            align2.append('-')
            i -= 1
        elif direction == 3:  # left (gap in seq1)
            align1.append('-')
            align2.append(seq2[j - 1])
            j -= 1
        else:
            break

    # We built the alignment backwards
    align1 = ''.join(reversed(align1))
    align2 = ''.join(reversed(align2))

    return max_score, align1, align2

if __name__=="__main__":
    Sequences=ReadFASTA(sys.argv[1])
    assert len(Sequences.keys())==2, "fasta file contains more than 2 sequences."
    seq1=Sequences[list(Sequences.keys())[0]]
    seq2=Sequences[list(Sequences.keys())[1]]

    score, align1, align2 = smith_waterman(seq1, seq2)

    print('Score: ', score)
    print('Seq1: ', align1)
    print('Seq2: ', align2)
