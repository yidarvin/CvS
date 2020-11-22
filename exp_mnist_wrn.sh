
# Prepare the Data
#python preproc/prepare_mnist.py --pData '/home/Data/MNIST'

# Train the Networks
#python train_mnist.py --pData '/home/Data/MNIST' --pModel '/home/Models' --numex 1 --lr 3e-4 --epoch 25 --name 'wrn'
#python train_mnist.py --pData '/home/Data/MNIST' --pModel '/home/Models' --numex 5 --lr 3e-4 --epoch 25 --name 'wrn'
#python train_mnist.py --pData '/home/Data/MNIST' --pModel '/home/Models' --numex 10 --lr 3e-4 --epoch 25 --name 'wrn'
#python train_mnist.py --pData '/home/Data/MNIST' --pModel '/home/Models' --numex 25 --lr 3e-4 --epoch 25 --name 'wrn'
#python train_mnist.py --pData '/home/Data/MNIST' --pModel '/home/Models' --numex 50 --lr 3e-4 --epoch 25 --name 'wrn'
#python train_mnist.py --pData '/home/Data/MNIST' --pModel '/home/Models' --numex 100 --lr 3e-4 --epoch 25 --name 'wrn'
python train_mnist.py --pData '/home/Data/MNIST' --pModel '/home/Models' --numex 250 --lr 3e-4 --epoch 100 --name 'wrn'
python train_mnist.py --pData '/home/Data/MNIST' --pModel '/home/Models' --numex 500 --lr 3e-4 --epoch 100 --name 'wrn'
python train_mnist.py --pData '/home/Data/MNIST' --pModel '/home/Models' --numex 1000 --lr 3e-4 --epoch 100 --name 'wrn'
python train_mnist.py --pData '/home/Data/MNIST' --pModel '/home/Models' --numex 10000 --lr 3e-4 --epoch 100 --name 'wrn'
