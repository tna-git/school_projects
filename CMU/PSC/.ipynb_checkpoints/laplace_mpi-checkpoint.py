from mpi4py import MPI
import numpy as np
import math

# --- Global sizes and variables ---
ROWS , COLUMNS = 1000 , 1000
MAX_TEMP_ERROR = 0.01

# --- Initialize MPI --- 
comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
size  = comm.Get_size()

# --- Decompose rows across ranks (block distribution) ---
rows_per = ROWS // size

# Global interior row index for this rank (1-based interior indexing)
# Rows 1-250, 251-500, 501-750, 751-1000
local_n = rows_per
start_i = 1 + rank * rows_per 
end_i   = start_i + rows_per - 1 

# --- Initialize temperature data ---
temperature      = np.zeros(( local_n + 2 , COLUMNS+2 ), dtype=float)
temperature_last = np.zeros(( local_n + 2 , COLUMNS+2 ), dtype=float)

# --- Set boundary columns (left/right) on all ranks ---
# left boundary j=0 is 0s (already zero)
# right boundary j=COLUMNS+1 is sine along i; only depends on global i
for li in range(1, local_n + 1): 
    gi = start_i + (li - 1)       # obtain global index (gi) from local row index (li)
    temperature_last[li, COLUMNS+1] = 100 * math.sin(((np.pi/2)/ROWS) * gi )
    temperature[li, COLUMNS+1] = temperature_last[li, COLUMNS+1]

# --- Set bottom boundary row on the last rank only ---
if rank == size - 1:
    # local row corresponding to global i = ROWS+1 (the bottom boundary)
    for j in range(COLUMNS + 1):  # 0..COLUMNS, last col handled above
        temperature_last[local_n + 1, j] = 100 * math.sin(((np.pi/2)/COLUMNS) * j )
        temperature[local_n + 1, j] = temperature_last[local_n + 1, j]
# Top (i=0) and left (j=0) are zeros already based on initialization 

# --- Defining neighbor ranks for ghost rows exchange ---
up    = rank - 1 if rank > 0 else MPI.PROC_NULL
down  = rank + 1 if rank < size - 1 else MPI.PROC_NULL

# --- Max iterations: read once on rank 0, broadcast ---
if rank == 0:
    max_iterations = int(input("Maximum iterations: "))
else:
    max_iterations = None
max_iterations = comm.bcast(max_iterations, root=0)

# --- Iteration loop ---
dt_global = 100
iteration = 1

while (dt_global > MAX_TEMP_ERROR) and (iteration < max_iterations):

    # 1) Halo exchange with temperature_last
    # send top interior row (li=1) upward, receive into top ghost (li=0)
    comm.Sendrecv(sendbuf=temperature_last[1, :],  dest=up,   sendtag=0,
                  recvbuf=temperature_last[0, :],  source=up, recvtag=1)
    # send bottom interior row (li=local_n) downward, receive into bottom ghost (li=local_n+1)
    comm.Sendrecv(sendbuf=temperature_last[local_n, :],      dest=down,   sendtag=1,
                  recvbuf=temperature_last[local_n + 1, :],  source=down, recvtag=0)

    # 2) Compute new interior temperature from temperature_last using vectorized method (faster than loop)
    temperature[1:-1, 1:-1] = 0.25 * (
        temperature_last[2:,   1:-1] +  # below
        temperature_last[:-2,  1:-1] +  # above
        temperature_last[1:-1, 2:  ] +  # right
        temperature_last[1:-1, :-2 ]    # left
    )

    # 3) Local convergence metric (use ABS difference)
    local_dt = np.max(np.abs(temperature[1:-1, 1:-1] - temperature_last[1:-1, 1:-1]))

    # 4) Swap so next iteration reads from updated field, keeping boundaries 
    temperature, temperature_last = temperature_last, temperature

    # 5) Global convergence check
    dt_global = comm.allreduce(local_dt, op=MPI.MAX)

    if rank == 0 and (iteration % 50 == 0 or dt_global <= MAX_TEMP_ERROR):
        print(f"iter {iteration}, dt={dt_global:.4g}")

    iteration += 1

# --- Graph of output to determine convergence ---
# Gather interior rows to rank 0 
send = temperature_last[1:local_n+1, :].copy()         # rows are contiguous
recv = None
if rank == 0:
    recv = np.empty((ROWS, COLUMNS+2), dtype=send.dtype)

comm.Gather(send, recv, root=0)                        # stacks by rank order

if rank == 0:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    full = np.zeros((ROWS+2, COLUMNS+2), dtype=recv.dtype)
    full[1:ROWS+1, :] = recv                           # place interior
    # top boundary already zeros; bottom/right were set by last rank / all ranks respectively

    fig, ax = plt.subplots()
    im = ax.imshow(full, origin="upper", aspect="auto")

    # Force the y-axis so row 0 is at the top
    ax.set_ylim(full.shape[0]-1, 0)   # e.g., 1001 → 0

    # Format nice ticks
    ax.set_yticks([0, 200, 400, 600, 800, 1000])
    ax.set_xticks([0, 200, 400, 600, 800, 1000])

    fig.colorbar(im, ax=ax)
    fig.savefig("plate_global.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
