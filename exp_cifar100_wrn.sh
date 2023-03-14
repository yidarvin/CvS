
# Prepare the Data
#python preproc/prepare_cifar100.py --pData /home/Data/CIFAR100

# Segmentation Propagation
##python preproc/segment_cifar100.py --pData '/home/Data/CIFAR100/cifar-100-python' --pSave '/home/Data/CIFAR100/cifar-100-smalldata-seg1' --pModel '/home/Models/cifar-10-smalldata-manualseg_1_default_late.pth'
python preproc/segment_cifar100.py --pData '/home/Data/CIFAR100/cifar-100-python' --pSave '/home/Data/CIFAR100/cifar-100-smalldata-seg10' --pModel '/home/Models/cifar-10-smalldata-manualseg_10_wrn_late.pth'
python preproc/segment_cifar100.py --pData '/home/Data/CIFAR100/cifar-100-python' --pSave '/home/Data/CIFAR100/cifar-100-smalldata-seg100' --pModel '/home/Models/cifar-10-smalldata-manualseg_100_wrn_late.pth'

# Train the Networks 1
#python train_cifar100.py --pData '/home/Data/CIFAR100/cifar-100-smalldata-seg1' --pVal '/home/Data/CIFAR100/cifar-100-python' --pModel '/home/Models' --numex 5 --lr 3e-5 --epoch 100
#python train_cifar100.py --pData '/home/Data/CIFAR100/cifar-100-smalldata-seg1' --pVal '/home/Data/CIFAR100/cifar-100-python' --pModel '/home/Models' --numex 10 --lr 3e-5 --epoch 100
#python train_cifar100.py --pData '/home/Data/CIFAR100/cifar-100-smalldata-seg1' --pVal '/home/Data/CIFAR100/cifar-100-python' --pModel '/home/Models' --numex 25 --lr 3e-5 --epoch 100
#python train_cifar100.py --pData '/home/Data/CIFAR100/cifar-100-smalldata-seg1' --pVal '/home/Data/CIFAR100/cifar-100-python' --pModel '/home/Models' --numex 50 --lr 3e-5 --epoch 100
#python train_cifar100.py --pData '/home/Data/CIFAR100/cifar-100-smalldata-seg1' --pVal '/home/Data/CIFAR100/cifar-100-python' --pModel '/home/Models' --numex 100 --lr 3e-5 --epoch 100
#python train_cifar100.py --pData '/home/Data/CIFAR100/cifar-100-smalldata-seg1' --pVal '/home/Data/CIFAR100/cifar-100-python' --pModel '/home/Models' --numex 500 --lr 3e-5 --epoch 100
#python train_cifar100.py --pData '/home/Data/CIFAR100/cifar-100-smalldata-seg1' --pVal '/home/Data/CIFAR100/cifar-100-python' --pModel '/home/Models' --numex 1000 --lr 3e-5 --epoch 100
#python train_cifar100.py --pData '/home/Data/CIFAR100/cifar-100-smalldata-seg1' --pVal '/home/Data/CIFAR100/cifar-100-python' --pModel '/home/Models' --numex 9999999999 --lr 3e-5 --epoch 100

# Train the Networks 10
##python train_cifar100.py --pData '/home/Data/CIFAR100/cifar-100-smalldata-seg10' --pVal '/home/Data/CIFAR100/cifar-100-python' --pModel '/home/Models' --numex 25 --lr 3e-4 --epoch 25 --name 'wrn'
##python train_cifar100.py --pData '/home/Data/CIFAR100/cifar-100-smalldata-seg10' --pVal '/home/Data/CIFAR100/cifar-100-python' --pModel '/home/Models' --numex 50 --lr 3e-4 --epoch 25 --name 'wrn'
python train_cifar100.py --pData '/home/Data/CIFAR100/cifar-100-smalldata-seg10' --pVal '/home/Data/CIFAR100/cifar-100-python' --pModel '/home/Models' --numex 100 --lr 3e-4 --epoch 50 --name 'wrn'
python train_cifar100.py --pData '/home/Data/CIFAR100/cifar-100-smalldata-seg10' --pVal '/home/Data/CIFAR100/cifar-100-python' --pModel '/home/Models' --numex 250 --lr 3e-4 --epoch 100 --name 'wrn'
python train_cifar100.py --pData '/home/Data/CIFAR100/cifar-100-smalldata-seg10' --pVal '/home/Data/CIFAR100/cifar-100-python' --pModel '/home/Models' --numex 1000 --lr 3e-4 --epoch 100 --name 'wrn'

# Train the Networks 100
python train_cifar100.py --pData '/home/Data/CIFAR100/cifar-100-smalldata-seg100' --pVal '/home/Data/CIFAR100/cifar-100-python' --pModel '/home/Models' --numex 250 --lr 3e-4 --epoch 100 --name 'wrn'
python train_cifar100.py --pData '/home/Data/CIFAR100/cifar-100-smalldata-seg100' --pVal '/home/Data/CIFAR100/cifar-100-python' --pModel '/home/Models' --numex 1000 --lr 3e-4 --epoch 100 --name 'wrn'
