
# Prepare the Data
python preproc/prepare_mnist.py --pData /home/Data/MNIST

# Train the Networks
python train_mnist.py --pData '/home/Data/MNIST' --pModel '/home/Models' --numex 1 --lr 3e-5 --epoch 1
python train_mnist.py --pData '/home/Data/MNIST' --pModel '/home/Models' --numex 5 --lr 3e-5 --epoch 1
python train_mnist.py --pData '/home/Data/MNIST' --pModel '/home/Models' --numex 10 --lr 3e-5 --epoch 1
python train_mnist.py --pData '/home/Data/MNIST' --pModel '/home/Models' --numex 25 --lr 3e-5 --epoch 1
python train_mnist.py --pData '/home/Data/MNIST' --pModel '/home/Models' --numex 50 --lr 3e-5 --epoch 1
python train_mnist.py --pData '/home/Data/MNIST' --pModel '/home/Models' --numex 100 --lr 3e-5 --epoch 1
python train_mnist.py --pData '/home/Data/MNIST' --pModel '/home/Models' --numex 300 --lr 3e-5 --epoch 1
python train_mnist.py --pData '/home/Data/MNIST' --pModel '/home/Models' --numex 1000 --lr 3e-5 --epoch 1
python train_mnist.py --pData '/home/Data/MNIST' --pModel '/home/Models' --numex 999999999 --lr 3e-5 --epoch 1
