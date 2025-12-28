
import numpy as np
import math
import matplotlib.pyplot as plt
from mpi4py import MPI
import time

s = time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

ROWS , COLUMNS = 1000 , 1000
MAX_TEMP_ERROR = 0.01
rows_per_rank = ROWS // size
start = (rank * rows_per_rank) + 1
end   = start + rows_per_rank - 1

temperature = np.zeros(( rows_per_rank+2 , COLUMNS+2 ))
temperature_last = np.zeros(( rows_per_rank+2 ,COLUMNS+2  ))




#Set right side boundary condition
for i in range(1, rows_per_rank+1):
    g = start + (i) - 1
    temperature_last[ i , COLUMNS+1 ] = 100 * math.sin( ( (3.14159/2) /ROWS) * g )
    temperature[ i , COLUMNS+1 ] = temperature_last[ i , COLUMNS+1 ]
#Set bottom boundary condition
if rank == size - 1:
    for j in range(COLUMNS+1):
        temperature_last[rows_per_rank+1 , j ] = 100 * math.sin( ( (3.14159/2) /COLUMNS) * j )
        temperature[rows_per_rank + 1, j] = temperature_last[rows_per_rank + 1, j]

if rank > 0:
    up = rank - 1
else:
    up = MPI.PROC_NULL

if rank < size-1:
    down = rank + 1
else:
    down = MPI.PROC_NULL       


if rank == 0:
    max_iterations = int (input("Maximum iterations: "))
else:
    max_iterations = None
max_iterations = comm.bcast(max_iterations, root=0)

dT = 100
iteration = 1

while ( dT > MAX_TEMP_ERROR ) and ( iteration < max_iterations):
    comm.Sendrecv(temperature_last[1,:], dest=up, sendtag=0,
                      recvbuf=temperature_last[0,:], source=up, recvtag=1)
    comm.Sendrecv(temperature_last[rows_per_rank,:], dest=down, sendtag=1,
                      recvbuf=temperature_last[rows_per_rank+1,:], source=down, recvtag=0)
    for i in range(1, rows_per_rank+1):
        for j in range(1, COLUMNS+1):
            temperature[i, j] = 0.25 * (
            temperature_last[i+1, j] +   # below
            temperature_last[i-1, j] +   # above
            temperature_last[i, j+1] +   # right
            temperature_last[i, j-1])

    for i in range(1, rows_per_rank+1):
        for j in range(1, COLUMNS+1):
            dt = max(dT, abs(temperature[i,j] - temperature_last[i,j]))
            temperature_last[i, j] = temperature[i, j]
    dt = np.max(np.abs(temperature[1:-1, 1:-1] - temperature_last[1:-1, 1:-1]))
    temperature, temperature_last = temperature_last, temperature
    dT = comm.allreduce(dt, op=MPI.MAX)
    #print(dt)
    if rank == 0:       
        print(iteration)
    iteration += 1
    #print(iteration)
send = temperature_last[1:rows_per_rank+1,:]
rec = None
if rank == 0:
    rec = np.empty((ROWS, COLUMNS+2), dtype=send.dtype)

comm.Gather(send,rec, root=0)
if rank == 0:
    p = np.zeros((ROWS+2 , COLUMNS+2))
    p[1:ROWS+1,:] = rec
           
    fig, ax = plt.subplots()
    im = ax.imshow(p, origin="upper", aspect="auto", cmap = "magma")
    ax.set_ylim(p.shape[0]-1, 0)
    ax.set_yticks(np.arange(0, ROWS+1, 200))
    ax.set_xticks(np.arange(0, ROWS+1, 200))
    fig.colorbar(im, ax=ax)
    fig.savefig("plate_parallelized.png", dpi=150)
    plt.close(fig)
e = time.time()

print(e-s, "seconds to run", ROWS, COLUMNS, max_iterations)


