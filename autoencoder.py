# By: LIU, Jiun Ee
# Codes modified from exercise 8 and 9 solutions

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Torchvision dataset
class FashionMNIST(datasets.MNIST):
    """ Fashion-MNIST dataset: https://github.com/zalandoresearch/fashion-mnist"""
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]

# define two dataset: for testing and training
dataset_train = FashionMNIST('./fashionMNIST', train=True, download=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

dataset_test = FashionMNIST('./fashionMNIST', train=False, download=False,
                       transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

print("Size of the training set: {}".format(dataset_train.train_data.size()))  
print("Type of the training set: {}".format(type(dataset_train.train_data)))
print("Size of the test set: {}".format(type(dataset_test.test_data))) 

# global parameters
IMAGE_SIZE  = 28
BATCH_SIZE  = 128
LR          = 0.001
WEIGHT_DECAY= 1e-5
NUM_EPOCHS  = 50
NUM_WORKERS = 2
USE_CUDA    = True

# define two data loader
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
test_loader  = torch.utils.data.DataLoader(dataset_test , batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

# define variables
images = torch.FloatTensor(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE)
if USE_CUDA:
    images = images.cuda()
images = Variable(images)
print('Autograd variable for images: type {}, size {}'.format(type(images.data), images.size()))

class Autoencoder(nn.Module):
    def __init__(self, input_shape=(IMAGE_SIZE, IMAGE_SIZE)):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            ########## YOUR CODE HERE #############
			nn.Linear(np.prod(input_shape),512),
			nn.ReLU(True),
			nn.Linear(512,256),
			nn.ReLU(True),
			nn.Linear(256,64),
			nn.Linear(256,64),
        )
        self.decoder = nn.Sequential(
            ########## YOUR CODE HERE #############
			nn.Linear(64,256),
			nn.ReLU(True),
                        nn.Linear(256,512),
                        nn.ReLU(True),
                        nn.Linear(512,np.prod(input_shape)),
                        nn.Tanh(), # mapping output to -1, 1 range
        )

    # mu and logvar: output of your encoder
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        ########## YOUR CODE HERE #############
	mu = self.encoder(x)
	logvar = self.encoder(x)
	x2 = self.reparameterize(mu, logvar) # I don't know how to incorporate mu and logvar as output from encoder
	x2 = self.decoder(x2)
        return x2, mu, logvar 

# VAE Loss
#
# recon_x: image reconstructions x: images mu and logvar: outputs of your encoder batch_size: batch_size img_size: width, respectively height of you images nc: number of image
# channels
def loss_function(recon_x, x, mu, logvar, batch_size, img_size, nc):
    MSE = F.mse_loss(recon_x, x.view(-1, img_size * img_size * nc))
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalize
    KLD /= batch_size * img_size * img_size * nc
    return MSE + KLD

# training routine
def train(model, optimizer, data_loader, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(data_loader):
		# reshape to vector
	        data = data.view(data.size(0), -1)
	        # move to cuda if available
        	if USE_CUDA:
            		data = data.cuda()
        	# convert to Variable
        	data = Variable(data)

	       # forward: evaluate with model
                output, mu, logvar = model(data)
                loss = loss_function(output, data, mu, logvar, BATCH_SIZE, IMAGE_SIZE, 3)

		# backward: compute gradient and update weights
		optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(data_loader.dataset),
                        100. * batch_idx / len(data_loader), loss.data[0]))


# testing routine
def test(show_plot=False):
    model.eval()
    test_loss = 0
    for i, (data, target) in enumerate(test_loader):
        # reshape to vector
        data = data.view(data.size(0), -1)
        # move to cuda if available
        if USE_CUDA:
            data = data.cuda()

        # convert to Variable
        data= Variable(data, volatile=True)
        # forward: evaluate with model
        output = model(data)
        test_loss += nn.MSELoss(size_average=False)(output, data).data[0] # sum up batch loss
        
        if show_plot == 1 and i == 0:
            inp = make_grid(to_img(data_in.cpu().data))
            out = make_grid(to_img(output.cpu().data))
            target = make_grid(to_img(data.cpu().data))
            plt.imshow(transforms.ToPILImage()(out))
            plt.axis('off')
            plt.show()
            plt.imshow(transforms.ToPILImage()(target))
            plt.axis('off')
            plt.show()
        
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
        
    return test_loss

model = Autoencoder()
if USE_CUDA:
        model = model.cuda()

# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

for epoch in range(1, NUM_EPOCHS + 1):
        reporttest = 0
        if epoch % 10 == 0:
                reporttest = 1

        train(model, optimizer, train_loader, epoch)

        loss_train, acc_train = test(model, optimizer, train_loader, reporttest)
        loss_test,  acc_test  = test(model, optimizer, test_loader, reporttest)

        log_baseline['loss_train'].append(loss_train)
        log_baseline['loss_test'].append(loss_test)
        log_baseline['acc_train'].append(acc_train)
        log_baseline['acc_test'].append(acc_test)

        if acc_test > best_acc:
                best_acc = acc_test
                torch.save(model.state_dict(), 'best_model_baseline.pth')

print('The best achieved accuracy: {:.2f}%\n'.format(best_acc))



# Manifold approximation using tSNE
#
# features: (numpy array) N x D feature matrix images: (numpy array) N x H x W x 3 path_save: string (path where you want to save the final image)
def apply_tnse_img(features, images, path_save='./tsne_img.png'):
    tnse = TSNE(n_components=2, init='pca', n_iter=1000, random_state=1254, perplexity=30, metric='euclidean')
    # np.set_printoptions(False)
    vis_data = tnse.fit_transform(features)
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]
    # get max heigth, width
    max_width = max([image.shape[0] for image in images])
    max_height = max([image.shape[1] for image in images])
    # get max, min coords
    x_min, x_max = vis_x.min(), vis_x.max()
    y_min, y_max = vis_y.min(), vis_y.max()
    # Fix the ratios
    res = 700
    sx = (x_max - x_min)
    sy = (y_max - y_min)
    if sx > sy:
        res_x = int(sx / float(sy) * res)
        res_y = res
    else:
        res_x = res
        res_y = int(sy / float(sx) * res)
    # impaint images
    canvas = np.ones((res_x + max_width, res_y + max_height, 3))
    x_coords = np.linspace(x_min, x_max, res_x)
    y_coords = np.linspace(y_min, y_max, res_y)
    for x, y, image in zip(vis_x, vis_y, images):
        w, h = image.shape[:2]
        x_idx = np.argmin((x - x_coords) ** 2)
        y_idx = np.argmin((y - y_coords) ** 2)
        try:
            canvas[x_idx:x_idx + w, y_idx:y_idx + h] = image
        except:
            print('Image out of borders.... skip!')
    # plot image
    fig = plt.figure()
    plt.imshow(canvas)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show(False)
    plt.pause(3)
    fig.savefig(path_save, bbox_inches='tight')
