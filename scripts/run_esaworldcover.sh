DATASET=/opt/data/benelux/benelux_sentinel2-rgb-median-2020_esa-world-cover
DATALOADER=S2_ESAWorldCover_DataLoader
TAG=esaworldcover


#python run.py --conf classicconfs.downsampl09 --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.001 --loss mse --batch_size 32 --epochs 100 --tag $TAG
#python run.py --conf kqmconfs.qmp03s1d           --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.0001 --loss mse --batch_size 32 --epochs 50 --tag $TAG
#python run.py --conf classicconfs.smvgg16_mse    --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.0001 --loss mse --batch_size 32 --epochs 50 --tag $TAG
#python run.py --conf classicconfs.unet04         --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.0001 --loss mse --batch_size 32 --epochs 50 --tag $TAG
#python run.py --conf classicconfs.smresnet18_mse --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.0001 --loss mse --batch_size 32 --epochs 50 --tag $TAG

#python run.py --conf classicconfs.downsampl09 --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.001  --loss pxce --batch_size 32 --epochs 50 --tag $TAG
#python run.py --conf classicconfs.smvgg16_mse --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.0001 --loss pxce --batch_size 32 --epochs 50 --tag $TAG
#python run.py --conf classicconfs.unet04      --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.0001 --loss pxce --batch_size 32 --epochs 50 --tag $TAG
#python run.py --conf classicconfs.smresnet18_mse --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.0001 --loss pxce --batch_size 32 --epochs 100 --tag $TAG


# ---- QMPatch exploration

#python run.py --conf kqmconfs.qmp06b           --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.0001 --loss mse --batch_size 32 --epochs 50 --tag $TAG
#python run.py --conf kqmconfs.qmp06a           --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.0001 --loss mse --batch_size 32 --epochs 50 --tag $TAG
#python run.py --conf kqmconfs.qmp04           --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.0001 --loss mse --batch_size 32 --epochs 50 --tag $TAG
#python run.py --conf kqmconfs.aeqm            --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.0001 --loss mse --batch_size 32 --epochs 50 --tag $TAG
python run.py --conf kqmconfs.qmp06aa           --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.0001 --loss mse --batch_size 32 --epochs 50 --tag $TAG
python run.py --conf kqmconfs.qmp03a           --dataset_folder $DATASET --dataloader_class $DATALOADER --learning_rate 0.0001 --loss mse --batch_size 32 --epochs 50 --tag $TAG







