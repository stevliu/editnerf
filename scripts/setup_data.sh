# Download rendered PhotoShapes chairs dataset.
mkdir -p data/photoshapes
cd data/photoshapes

wget editnerf.csail.mit.edu/photoshapes.zip
echo 'Unzipping photoshapes dataset'
unzip -q photoshapes.zip
mv photoshapes/* .

cd ../../

# Download and process rendered CARLA cars dataset.
mkdir -p data/carla/carla_images
cd data/carla/carla_images

wget https://s3.eu-central-1.amazonaws.com/avg-projects/graf/data/carla.zip
echo 'Unzipping CARLA dataset'
unzip -q carla.zip

cd ..
wget https://s3.eu-central-1.amazonaws.com/avg-projects/graf/data/carla_poses.zip
unzip -q carla_poses.zip

cd ../../
echo 'Formatting CARLA dataset'
python utils/setup_cars.py

# Download and process rendered Dosovitskiy chairs dataset. 
mkdir -p data/dosovitskiy_chairs/
cd data/dosovitskiy_chairs

wget https://www.di.ens.fr/willow/research/seeing3Dchairs/data/rendered_chairs.tar
echo 'Unzipping Dosovitskiy chairs dataset'
tar -xf rendered_chairs.tar
cd ../../

echo 'Formatting Dosovitskiy chairs dataset'
python utils/setup_dosovitskiy.py
