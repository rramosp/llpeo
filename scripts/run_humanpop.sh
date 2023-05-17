DATASET=/opt/data/benelux/benelux_sentinel2-rgb-median-2020_humanpop2015
DATALOADER=S2_HumanPopulation_DataLoader
TAG=humanpop


#python run.py --conf classicconfs.downsampl09 --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.001 --loss mse --batch_size 8 --epochs 50 --tag $TAG
#python run.py --conf kqmconfs.qmp03s1d           --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.0001 --loss mse --batch_size 8 --epochs 50 --tag $TAG
#python run.py --conf classicconfs.smvgg16_mse    --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.0001 --loss mse --batch_size 8 --epochs 50 --tag $TAG
#python run.py --conf classicconfs.unet04         --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.0001 --loss mse --batch_size 8 --epochs 50 --tag $TAG
#python run.py --conf classicconfs.smresnet18_mse --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.0001 --loss mse --batch_size 8 --epochs 50 --tag $TAG

#python run.py --conf classicconfs.downsampl09 --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.001  --loss pxce --batch_size 8 --epochs 50 --tag $TAG
#python run.py --conf classicconfs.smvgg16_mse --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.0001 --loss pxce --batch_size 8 --epochs 50 --tag $TAG
#python run.py --conf classicconfs.unet04      --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.0001 --loss pxce --batch_size 8 --epochs 50 --tag $TAG
#python run.py --conf classicconfs.smresnet18_mse --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.0001 --loss pxce --batch_size 8 --epochs 50 --tag $TAG

# exploration con kqm
#python run.py --conf kqmconfs.qmp03           --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.0001 --loss mse --batch_size 8 --epochs 50 --tag $TAG
#python run.py --conf kqmconfs.qmp06b           --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.0001 --loss mse --batch_size 8 --epochs 50 --tag $TAG
#python run.py --conf kqmconfs.qmp06a           --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.001 --loss mse --batch_size 8 --epochs 50 --tag $TAG
#python run.py --conf kqmconfs.qmp04           --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.0001 --loss mse --batch_size 8 --epochs 50 --tag $TAG
python run.py --conf kqmconfs.qmp06aa           --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.0001 --loss mse --batch_size 8 --epochs 50 --tag $TAG
python run.py --conf kqmconfs.qmp03a           --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.0001 --loss mse --batch_size 8 --epochs 50 --tag $TAG


