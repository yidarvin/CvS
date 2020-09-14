
# Prepare the Data
#python preproc/prepare_cifar10.py --pData /home/Data/CIFAR10

# Train the Networks
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-manualseg' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 1 --lr 3e-4 --epoch 100
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-manualseg' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 5 --lr 3e-4 --epoch 100
##python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-manualseg' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 10 --lr 1e-1 --epoch 100
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-manualseg' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 25 --lr 3e-4 --epoch 100
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-manualseg' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 50 --lr 3e-4 --epoch 100
python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-manualseg' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 100 --lr 1e-1 --epoch 100

# Segmentation Propagation
#python preproc/segment_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-batches-py' --pSave '/home/Data/CIFAR10/cifar-10-smalldata-seg1' --pModel '/home/Models/cifar-10-smalldata-manualseg_1_default_late.pth'
##python preproc/segment_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-batches-py' --pSave '/home/Data/CIFAR10/cifar-10-smalldata-seg10' --pModel '/home/Models/cifar-10-smalldata-manualseg_10_default_best.pth'
python preproc/segment_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-batches-py' --pSave '/home/Data/CIFAR10/cifar-10-smalldata-seg100' --pModel '/home/Models/cifar-10-smalldata-manualseg_100_default_late.pth'

# Train the Networks 1
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg1' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 5 --lr 3e-5 --epoch 100
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg1' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 10 --lr 3e-5 --epoch 100
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg1' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 25 --lr 3e-5 --epoch 100
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg1' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 50 --lr 3e-5 --epoch 100
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg1' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 100 --lr 3e-5 --epoch 100
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg1' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 500 --lr 3e-5 --epoch 100
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg1' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 1000 --lr 3e-5 --epoch 100
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg1' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 9999999999 --lr 3e-5 --epoch 100

# Train the Networks 10
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg10' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 25 --lr 3e-4 --epoch 100
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg10' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 50 --lr 3e-4 --epoch 100
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg10' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 100 --lr 3e-4 --epoch 100
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg10' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 250 --lr 3e-4 --epoch 100
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg10' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 500 --lr 3e-4 --epoch 100
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg10' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 1000 --lr 3e-4 --epoch 100
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg10' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 2500 --lr 3e-4 --epoch 100
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg10' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 5000 --lr 3e-4 --epoch 100

# Train the Networks 100
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg100' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 250 --lr 3e-4 --epoch 100
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg100' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 500 --lr 3e-4 --epoch 100
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg100' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 1000 --lr 3e-4 --epoch 100
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg100' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 2500 --lr 3e-4 --epoch 100
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg100' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 5000 --lr 3e-4 --epoch 100
