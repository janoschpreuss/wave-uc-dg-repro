 #!/bin/bash

killall -9 python3
python3 precond-1d-comp.py 1
killall -9 python3
python3 precond-1d-comp.py 2
killall -9 python3
python3 noGCC-restricted-1d.py
killall -9 python3

