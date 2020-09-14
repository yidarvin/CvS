
# Prepare the Data
#python preproc/prepare_cifar10.py --pData /home/Data/CIFAR10

# Train the Networks
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-manualseg' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 1 --lr 3e-5 --epoch 100
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-manualseg' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 5 --lr 3e-5 --epoch 100
python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-manualseg' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 10 --lr 1e-1 --epoch 100 --pretrained 1 --name 'pretrained'
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-manualseg' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 25 --lr 3e-5 --epoch 100
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-manualseg' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 50 --lr 3e-5 --epoch 100
##python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-manualseg' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 100 --lr 3e-5 --epoch 100 --pretrained 1 --name 'pretrained'

# Segmentation Propagation
#python preproc/segment_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-batches-py' --pSave '/home/Data/CIFAR10/cifar-10-smalldata-seg1pretrained' --pModel '/home/Models/cifar-10-smalldata-manualseg_1_pretrained_late.pth'
python preproc/segment_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-batches-py' --pSave '/home/Data/CIFAR10/cifar-10-smalldata-seg10pretrained' --pModel '/home/Models/cifar-10-smalldata-manualseg_10_pretrained_late.pth'
##python preproc/segment_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-batches-py' --pSave '/home/Data/CIFAR10/cifar-10-smalldata-seg100pretrained' --pModel '/home/Models/cifar-100-smalldata-manualseg_100_pretrained_late.pth'

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
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg10pretrained' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 25 --lr 3e-5 --epoch 30 --pretrained 1 --name 'pretrained'
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg10pretrained' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 50 --lr 3e-5 --epoch 30 --pretrained 1 --name 'pretrained'
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg10pretrained' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 100 --lr 3e-5 --epoch 30 --pretrained 1 --name 'pretrained'
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg10pretrained' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 250 --lr 3e-5 --epoch 30 --pretrained 1 --name 'pretrained'
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg10pretrained' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 500 --lr 3e-5 --epoch 30 --pretrained 1 --name 'pretrained'
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg10pretrained' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 1000 --lr 3e-5 --epoch 30 --pretrained 1 --name 'pretrained'
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg10pretrained' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 2500 --lr 3e-5 --epoch 30 --pretrained 1 --name 'pretrained'
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg10pretrained' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 5000 --lr 3e-5 --epoch 30 --pretrained 1 --name 'pretrained'

# Train the Networks 100
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg100pretrained' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 250 --lr 3e-5 --epoch 100 --pretrained 1 --name 'pretrained'
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg100pretrained' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 500 --lr 3e-5 --epoch 100 --pretrained 1 --name 'pretrained'
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg100pretrained' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 1000 --lr 3e-5 --epoch 100 --pretrained 1 --name 'pretrained'
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg100pretrained' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 2500 --lr 3e-5 --epoch 100 --pretrained 1 --name 'pretrained'
#python train_cifar10.py --pData '/home/Data/CIFAR10/cifar-10-smalldata-seg100pretrained' --pVal '/home/Data/CIFAR10/cifar-10-batches-py' --pModel '/home/Models' --numex 5000 --lr 3e-5 --epoch 100 --pretrained 1 --name 'pretrained'
