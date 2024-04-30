 #!/bin/bash

killall -9 python3
python3 precond-3d-comp.py 1 1
killall -9 python3
python3 precond-3d-comp.py 2 1
killall -9 python3
python3 precond-3d-comp.py 3 1
killall -9 python3
python3 precond-3d-comp.py 1 0
killall -9 python3
python3 precond-3d-comp.py 2 0
killall -9 python3
python3 precond-3d-comp.py 3 0
killall -9 python3
