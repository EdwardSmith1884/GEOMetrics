from utils import *
from layers import * 
from models import *
from voxel  import voxel2obj
from torch.utils.data import DataLoader

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=40, help='Random seed.')
parser.add_argument('--epochs', type=int, default=150,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Initial learning rate.')
parser.add_argument('--exp_id', type=str, default='Test',
                    help='The experiment name')


args = parser.parse_args()
batch_size = 16
latent_length = 50
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# data settings
images = 'data/images/*'
voxels = 'data/voxels/'
meshes = 'data/mesh_info/'
checkpoint_dir = "checkpoint/" +  args.exp_id +'/'
save_dir =  "plots/" +  args.exp_id +'/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# load data
train_data = Voxel_loader(images, meshes, voxels, set_type = 'train')
train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=8, collate_fn = train_data.collate)

valid_data = Voxel_loader(images, meshes, voxels, set_type = 'valid')
valid_loader = DataLoader(valid_data, batch_size=16, shuffle=False, num_workers=8, collate_fn = valid_data.collate)

# load models
encoder_mesh = MeshEncoder(latent_length)
decoder = Decoder(latent_length)

# pytorch it up 
decoder.cuda(), encoder_mesh.cuda()
params = list(decoder.parameters()) + list(encoder_mesh.parameters())   
optimizer = optim.Adam(params,lr=args.lr)


class Engine() :
    def __init__(self): 
        self.best = 1000
        self.epoch = 0 
        self.train_losses = []
        self.valid_losses = []

    def train(self):
        decoder.train(),  encoder_mesh.train()
        total_loss = 0
        iteration = 0 
        for batch in tqdm(train_loader): 
            voxel_gt = batch['voxels'].cuda()
            optimizer.zero_grad()    
            
            latent = None
            # pass through encoder
            for mesh, adj in zip(batch['verts'], batch['adjs']):
                mesh = mesh.cuda()
                adj = adj.cuda()
                if latent is None: 
                    latent = encoder_mesh(mesh, adj).unsqueeze(0)
                else: latent = torch.cat((latent,encoder_mesh(mesh,adj).unsqueeze(0))) 
          
          	# pass trough decoder 
            voxel_pred = decoder(latent)

            # calculate loss and optimize with respect to it 
            loss = torch.mean((voxel_pred- voxel_gt)**2 )
            loss.backward()
            optimizer.step()
            track_loss = loss.item()
            total_loss += track_loss

            # print info occasionally 
            if iteration % 20 ==0 : 
                message = f'Train Loss: Epoch: {self.epoch}, loss: {track_loss}, best: {self.best}'
                tqdm.write(message)
            iteration += 1 

        self.train_losses.append(total_loss / float(iteration))

   


    def validate(self): 
        decoder.eval(), encoder_mesh.eval()
        total_loss = 0
        iteration = 0 
        for batch in tqdm(valid_loader): 
            voxel_gt = batch['voxels'].cuda()
            optimizer.zero_grad()    
            
            latent = None
            # pass through encoder
            for mesh, adj in zip(batch['verts'], batch['adjs']):
                mesh = mesh.cuda()
                adj = adj.cuda()
                if latent is None: 
                    latent = encoder_mesh(mesh, adj).unsqueeze(0)
                else: latent = torch.cat((latent,encoder_mesh(mesh,adj).unsqueeze(0))) 
          
            # pass trough decoder 
            voxel_pred = decoder(latent)

            # calculate loss and optimize with respect to it 
            loss = torch.mean((voxel_pred- voxel_gt)**2 )
            track_loss = loss.item()
            total_loss += track_loss

            # print info occasionally 
            if iteration % 20 ==0 : 
                message = f'Valid Loss: Epoch: {self.epoch}, new: {total_loss / float(iteration + 1 )}, cur: {self.best}'
                tqdm.write(message)
            iteration += 1 

        self.valid_losses.append(total_loss / float(iteration))
      
          
    def save(self): 
        if self.valid_losses[-1] <= self.best:
            self.best = self.valid_losses[-1] 
            torch.save(decoder.state_dict(), checkpoint_dir + 'decoder')
            torch.save(encoder_mesh.state_dict(), checkpoint_dir + 'encoder')







trainer = Engine()



for epoch in range(args.epochs):
    trainer.epoch = epoch
    trainer.train()
    trainer.validate()
    trainer.save()

print ('Saving latent codes of all models')
encoder_mesh.load_state_dict(torch.load(checkpoint_dir + 'encoder'))
encoder_mesh.eval()
if not os.path.exists('data/latent/'):
    os.makedirs('data/latent/')
for batch in tqdm(train_loader):
    for v, a, n  in zip(batch['verts'], batch['adjs'], batch['names']):
        
        latent = encoder_mesh(v.cuda(), a.cuda() )
        np.save('data/latent/' + n + '_latent', latent.data.cpu().numpy())

        
     
