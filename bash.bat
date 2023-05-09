cd C:\Diffusion_Thesis\cin_256
conda activate DSD
python distill.py -m cin -t DSDI -u 4000 -lr 0.0000001
python distill.py -m cin -t DSDGEXP -u 4000 -lr 0.0000001
python distill.py -m cin -t DSDGL -u 4000 -lr 0.0000001
python distill.py -m cin -t DSDN -u 4000 -lr 0.0000001
python distill.py -m cin -t DSDI -u 5000 -lr 0.00000006
python distill.py -m cin -t DSDGEXP -u 5000 -lr 0.00000006
python distill.py -m cin -t DSDGL -u 5000 -lr 0.00000006
python distill.py -m cin -t DSDN -u 5000 -lr 0.00000006