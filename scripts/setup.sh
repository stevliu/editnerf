#Pull the models
wget editnerf.csail.mit.edu/weights.zip
unzip weights.zip
cp -r weights/* ui/

mkdir -p logs
cp -r weights/* logs/

#Pull what each instance looks like
wget editnerf.csail.mit.edu/instances.zip
unzip instances.zip
mv instances/photoshapes/ ui/photoshapes/instances
mv instances/dosovitskiy_chairs/ ui/dosovitskiy_chairs/instances
mv instances/cars/ ui/cars/instances

#Pull the editing data
wget editnerf.csail.mit.edu/examples.zip
unzip examples.zip
mv examples/photoshapes/* ui/photoshapes
mv examples/dosovitskiy_chairs/* ui/dosovitskiy_chairs
mv examples/cars/* ui/cars
mv examples/real_chair/* ui/dosovitskiy_chairs/real_chair