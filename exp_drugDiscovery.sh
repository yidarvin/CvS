
# Prepare the Data
#DOWNLOAD THE DATA FROM BOX.
#PLACE TRAIN/VAL SPLIT IN A FOLDER.

# Train the Networks
#python train_drugDiscoveryBinary.py --pData '/home/darvin/Data/drugDiscovery/train000' --pModel '/home/darvin/Models' --lr 3e-4 --epoch 100 --name 'sanityCheck'
#python train_drugDiscoveryBinary.py --pData '/home/darvin/Data/drugDiscovery/train001' --pModel '/home/darvin/Models' --lr 3e-4 --epoch 100 --name 'noisyImages'

#python inference_drugDiscoveryBinary.py --pData '/home/darvin/Data/drugDiscovery/exp000/U18166A' --pModel '/home/darvin/Models' --name 'sanityCheck'
#python inference_drugDiscoveryBinary.py --pData '/home/darvin/Data/drugDiscovery/exp000/U18166A' --pModel '/home/darvin/Models' --name 'noisyImages'

python inference_drugDiscoveryBinary.py --pData '/home/darvin/Data/drugDiscovery/exp000/wt_val_noisy' --pModel '/home/darvin/Models' --name 'sanityCheck'
