
# Prepare the Data
#DOWNLOAD THE DATA FROM BOX.
#PLACE TRAIN/VAL SPLIT IN A FOLDER.

# Train the Networks
python train_drugDiscoveryBinary.py --pData '/home/darvin/Data/drugDiscvoery/train000' --pModel '/home/darvin/Models' --lr 3e-4 --epoch 100
