from utils import *
from layers import * 
from models import *

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=40, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Initial learning rate.')
parser.add_argument('--object', type=str, default='chair',
                    help='The object we are learning from ')
args = parser.parse_args()
batch_size = 16
latent_length = 50
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# data settings
images = 'data/images/' + args.object +'/'
voxels = 'data/voxels/' + args.object +'/'
meshes = 'data/mesh_info/' +  args.object +'/'
checkpoint_dir = "checkpoint/" +  args.object +'/'
save_dir =  "plots/" +  args.object +'/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# Load data
data_train = Voxel_loader(images + 'train/' ,meshes, voxels)
data_valid = Voxel_loader(images + 'valid/' ,meshes, voxels)
# load models
encoder_mesh = MeshEncoder(latent_length)
decoder = Decoder(latent_length)

# pytorch it up 
decoder.cuda(), encoder_mesh.cuda()
params = list(decoder.parameters()) + list(encoder_mesh.parameters())   
optimizer = optim.Adam(params,lr=args.lr)


# training interation 
def train(loader, epoch, num_interations, iteration, best):
    t = time()
    decoder.train(),  encoder_mesh.train()
    optimizer.zero_grad()    
    batch = loader.load_batch(batch_size)
    latent = None

    # pass through encoder
    for mesh, adj, name in zip(batch['verts'], batch['adjs'], batch['names']):
        if latent is None: 
            latent = encoder_mesh(mesh,adj ).view(1,latent_length)
        else: latent = torch.cat((latent,encoder_mesh(mesh,adj).view(1,latent_length)), dim = 0) 
  
  	# pass trough decoder 
    voxel_pred = decoder(latent)

    # calculate loss and optimize with respect to it 
    loss = torch.mean((voxel_pred- batch['voxels'])**2 )
    full_loss = loss 
    full_loss.backward()
    optimizer.step()
    track_loss = float(loss.data.cpu().numpy()) 

    # print info occasionally 
    if iteration % 10 != 0 : return track_loss
    print 'Epoch: {:d}'.format(epoch),
    print ' Iter: {:d}/{:d}'.format(iteration, num_interations),  
    print ' loss: {:.4f}'.format(track_loss),
    print ' time: {:.2f}s'.format(time() - t),
    print ' best: {:.4f}'.format(best)

    return track_loss

# validation iteration
def val_batch(verts, adj, voxels, imgs): 
    latent = None 
    for mesh, adj,  im in zip(verts, adj, imgs): 
        copy_mesh = mesh.clone()
        copy_adj = {}
        copy_adj['adj'] = adj['adj'].clone()
  
        if latent is None: 
            latent = encoder_mesh(copy_mesh,copy_adj ).view(1,latent_length)
        else: latent = torch.cat((latent,encoder_mesh(copy_mesh,copy_adj).view(1,latent_length)), dim = 0)
        latent.detach()
       

    voxel_pred = decoder(latent)
    latent = None 
    loss = torch.mean((voxel_pred- voxels)**2)
    loss =float(loss.data.cpu().numpy()) 
    return loss


def validate(batch): 
    decoder.eval(), encoder_mesh.eval()
    optimizer.zero_grad()    
    valid_loss = 0 
    latent = None
    from tqdm import tqdm 
    for i in (range(len(batch['verts'])//1)): 
        valid_loss += val_batch(batch['verts'][i*1:i*1 + 1], batch['adjs'][i*1:i*1 + 1], batch['voxels'][i*1:i*1 + 1], batch['imgs'][i*1:i*1 + 1])
    
    valid_loss /= float(i+1)
          
            

    return valid_loss




full_valid_loss, full_train_loss, length, min_val = [],[], 100, 1000
valid_batch = data_valid.load_batch(19, ordered = True, images = True )
for epoch in range(args.epochs):
    for i in range(length):  
        full_train_loss.append(train(data_train, epoch, length, i, min_val))
    full_valid_loss.append(validate(valid_batch))

    if full_valid_loss[-1] < min_val: 
        min_val = full_valid_loss[-1]
        torch.save(decoder.state_dict(), checkpoint_dir + 'decoder')
        torch.save(encoder_mesh.state_dict(), checkpoint_dir + 'encoder')
     
    if epoch>0: 
        graph(save_dir, full_train_loss[1*length:], full_valid_loss[1:])
        
     
