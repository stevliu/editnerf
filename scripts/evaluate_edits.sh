## Compute metrics on all the editing results. 
python utils/evaluate_edits.py --dataset photoshapes --expdir ui/photoshapes/edit_0/ --src 0 --tgt 328
python utils/evaluate_edits.py --dataset photoshapes --expdir ui/photoshapes/edit_1_part2/ --src 0 --tgt 287
python utils/evaluate_edits.py --dataset photoshapes --expdir ui/photoshapes/edit_2_part2/ --src 5 --tgt 253

python utils/evaluate_edits.py --dataset dosovitskiy_chairs --expdir ui/dosovitskiy_chairs/57/ --src 57 --tgt 663 
python utils/evaluate_edits.py --dataset dosovitskiy_chairs --expdir ui/dosovitskiy_chairs/21/ --src 21 --tgt 562
python utils/evaluate_edits.py --dataset dosovitskiy_chairs --expdir ui/dosovitskiy_chairs/259/ --src 259 --tgt 883