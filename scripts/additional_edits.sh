cd ui 

# Additional color edits
python editingapp.py --config photoshapes/config.txt --instance 0 --video --editname seat2red
python editingapp.py --config photoshapes/config.txt --instance 5 --video --editname seat2purple
python editingapp.py --config cars/config.txt --instance 0 --video --editname green2pink   
python editingapp.py --config cars/config.txt --instance 8 --video --editname pink2blue

# Additional shape edits
python editingapp.py --config photoshapes/config.txt --instance 1 --video --editname removebar
python editingapp.py --config photoshapes/config.txt --instance 1 --video --editname removearms
python editingapp.py --config photoshapes/config.txt --instance 7 --video --editname greyremovearm
python editingapp.py --config photoshapes/config.txt --instance 7 --video --editname removewheels

cd ..