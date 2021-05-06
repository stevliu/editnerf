cd ui

## Color edits
# Change the color of one part
python editingapp.py --config photoshapes/config.txt --instance 0 --editname edit_0

# Make one color edit, then make another color edit
python editingapp.py --config photoshapes/config.txt --instance 0 --editname edit_1_part1 --second_editname edit_1_part2
python editingapp.py --config photoshapes/config.txt --instance 5 --editname edit_2_part1 --second_editname edit_2_part2

## Shape edits
python editingapp.py --config dosovitskiy_chairs/config.txt --instance 57 --editname 57
python editingapp.py --config dosovitskiy_chairs/config.txt --instance 21 --editname 21
python editingapp.py --config dosovitskiy_chairs/config.txt --instance 259 --editname 259

## Color and shape transfer
python editing_utils.py --config photoshapes/config.txt 

## Real Image Editing
python editingapp.py --config dosovitskiy_chairs/config.txt --expname dosovitskiy_chairs/real_chair --instance 0 --editname removelegs --video

cd ..
