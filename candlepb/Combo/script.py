from mpi4py import MPI
import os
import socket
import subprocess


hostname = socket.gethostname()
rank = MPI.COMM_WORLD.Get_rank()
gpu_device = rank % 2
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)



arg = ' --id='+str(rank)+' --fname=best_archs.json' +'  &> ' + 'output/part'+ str(rank)+'.out'
cmd = "python combo_posttraining.py " + arg 
print(cmd)
subprocess.run(cmd, shell=True)

