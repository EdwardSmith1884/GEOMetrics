from utils import *
from layers import *
from models import *

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--eval', action='store_true', default=False,
                    help='Evaulate F1 score')
parser.add_argument('--eval_vis', action='store_true', default=False,
                    help='Evaulate mesh predictions')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Evaulate with pretrained model')

parser.add_argument('--seed', type=int, default=41, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Initial learning rate.')
parser.add_argument('--best_accuracy', action='store_true', default=False,
                    help='Train to achieve highest accuracy')
parser.add_argument('--latent_loss', action='store_true', default=False,
                    help='Train with latent loss')
parser.add_argument('--exp_id', type=str, default='Test',
                    help='The experiment name')

args = parser.parse_args()
sample_num = 3000

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# establish directories
images = 'data/images/*'
meshes = 'data/mesh_info/'
samples =  'data/surfaces/'

checkpoint_dir = "checkpoint/" +  args.exp_id +'/'
save_dir =  "plots/" +  args.exp_id +'/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# gather data
adj_info, initial_positions= load_initial('482.obj')
initial_positions = Variable(initial_positions.cuda())
num_verts = initial_positions.shape[0]

# load training set 
train_data = Mesh_loader(images, meshes, samples, set_type = 'train', sample_num = sample_num)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=8, collate_fn = train_data.collate)
classes = ['bench','cabinet','car','cellphone','chair','lamp','monitor','plane','rifle','sofa','speaker','table','watercraft']

# load validation set 
valid_loaders = []
for c in classes:
    valid_images = 'data/images/' + c 
    valid_data = Mesh_loader(valid_images, meshes, samples, set_type = 'valid', sample_num = sample_num, num= 0)
    valid_loaders.append(DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=8, collate_fn = valid_data.collate))



# initialize models
encoder1 = VGG()
encoder2 = VGG()
encoder3 = VGG()
modelA = BatchMeshDeformationBlock(963, num_verts)
modelB = BatchMeshDeformationBlock(1155, num_verts)
modelC = BatchMeshDeformationBlock(1155, num_verts)
encoder_mesh = BatchMeshEncoder(50)
modelA.cuda(), encoder1.cuda(), encoder2.cuda(),encoder3.cuda(), modelB.cuda(), modelC.cuda(), encoder_mesh.cuda()

params = list(modelA.parameters()) + list(encoder1.parameters()) + list(encoder2.parameters()) + list(encoder3.parameters()) +list(modelB.parameters())  +list(modelC.parameters())
optimizer = optim.Adam(params,lr=args.lr)



if args.latent_loss: 
    encoder_mesh.load_state_dict(torch.load(checkpoint_dir + 'encoder'))
    encoder_mesh.eval()





class Engine(): 
    def __init__(self): 
        self.epoch = 0 
        self.best_f1 = 0
        self.best_ptp = 100000
        self.train_loss = []
        self.train_f1 = []
        self.valid_loss_ptp = []
        self.valid_loss_f1 = []

    def train(self):
        total_loss = 0 
        total_f1 = 0 
        iterations = 0 
        modelA.train(), modelB.train(), modelC.train(), encoder1.train(), encoder2.train(), encoder3.train()

        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            
            #  initialize data   
            images = batch['imgs'].cuda()
            gt_samples = batch['samples'].cuda()
            train_latent = batch['latent'].cuda()
            on_latent = batch['encode'].cuda()
            img_info = batch['img_info'].cuda()
            batch_size  = images.shape[0]
            initial_positions_batch = initial_positions.unsqueeze(0).expand(batch_size, num_verts, 3)

            # extract image features
            features1  = encoder1(images)
            features2  = encoder2(images)
            features3  = encoder3(images)

            # first prediction
            vertex_features = batched_pooling(features1, initial_positions_batch, img_info.clone())
            vertex_features, vertex_positions_1 = modelA(initial_positions_batch, vertex_features, adj_info['adj'])
            vertex_positions_1 = initial_positions_batch + vertex_positions_1

            # second prediction
            vertex_features = torch.cat((vertex_features, batched_pooling(features2, vertex_positions_1.clone(), img_info.clone())), dim=-1)
            vertex_features, vertex_positions_2 = modelB( vertex_positions_1.clone(), vertex_features, adj_info['adj'])
            vertex_positions_2 = vertex_positions_2 + vertex_positions_1

            # third prediction 
            vertex_features = torch.cat((vertex_features, batched_pooling(features3, vertex_positions_2.clone(), img_info.clone())), dim=-1 )
            _,vertex_positions_3 = modelC(vertex_positions_2.clone(),vertex_features, adj_info['adj'])
            vertex_positions_3 = vertex_positions_3 + vertex_positions_2    
      
            # surface loss
            if not args.best_accuracy: 
                surface_loss_1 = batch_point_to_surface(vertex_positions_1.clone(), adj_info, gt_samples, num = sample_num)
                surface_loss_2 = batch_point_to_surface(vertex_positions_2.clone(), adj_info, gt_samples, num = sample_num)
                surface_loss_3, f1 = batch_point_to_surface(vertex_positions_3.clone(), adj_info, gt_samples, num = sample_num, f1 = True )
                surface_loss = surface_loss_1*.2 + surface_loss_2*.2 + surface_loss_3*2
                total_loss += surface_loss_3 
            else: 
                surface_loss, f1 = batch_point_to_surface(vertex_positions_3.clone(), adj_info, gt_samples, num = sample_num, f1 = True )
                total_loss += surface_loss 
            total_f1 += f1  

            # edge_loss 
            if not args.best_accuracy:
                edge_loss = batch_calc_edge(vertex_positions_1.clone(), adj_info) * 300
                edge_loss += batch_calc_edge(vertex_positions_2.clone(), adj_info) * 300
                edge_loss += batch_calc_edge(vertex_positions_3.clone(), adj_info) * 300
            else: 
                edge_loss = torch.FloatTensor([0]).cuda() 
            

            # lap loss
            if not args.best_accuracy:
                lap_loss_1  = torch.mean(torch.sum((batch_get_lap_info( initial_positions, adj_info) - batch_get_lap_info( vertex_positions_1, adj_info))**2, 2 ))* 1500
                lap_loss_2  = torch.mean(torch.sum((batch_get_lap_info( vertex_positions_1, adj_info) - batch_get_lap_info( vertex_positions_2, adj_info))**2, 2)) * 1500
                lap_loss_2  += torch.mean(torch.sum((vertex_positions_1 - vertex_positions_2)**2, 2)) * 100
                lap_loss_3  = torch.mean(torch.sum((batch_get_lap_info( vertex_positions_2, adj_info) - batch_get_lap_info( vertex_positions_3, adj_info))**2, 2)) * 1500
                lap_loss_3  += torch.mean(torch.sum((vertex_positions_2- vertex_positions_3)**2, 2)) * 100
                lap_loss  = .2*(lap_loss_1*.3  + lap_loss_2  + lap_loss_3)
            else:
                lap_loss = torch.FloatTensor([0]).cuda() 

            if args.latent_loss and (on_latent.sum() != 0): 
                latent_pred = encoder_mesh(vertex_positions_3, adj_info['adj'])
                latent_loss = .0005*(torch.mean(torch.abs(latent_pred - train_latent), dim = 1) * on_latent / ( on_latent.sum())).sum()
            else: 
                latent_loss = torch.FloatTensor([0]).cuda() 

            loss = edge_loss  + surface_loss + lap_loss + latent_loss
                    
            loss.backward()
            optimizer.step()

            message = f'Train || Epoch: {self.epoch}, loss: {loss.item():.2f}, surf: {surface_loss.item():.2f}, '
            message += f'lt:{latent_loss.item():.2f}, e: {edge_loss.item():.2f}, lp: {lap_loss.item():.2f}, f1 :{f1:.2f}, b_ptp: {self.best_ptp:.2f}, b_f1:  {self.best_f1:.2f}'
            tqdm.write(message)
            iterations += 1.


        self.train_loss.append(total_loss / iterations)
        self.train_f1.append(total_f1 / iterations)
        

        
    def validate(self): 
        total_loss_ptp = 0 
        total_loss_f1 = 0 
        
        modelA.eval(), modelB.eval(), modelC.eval(), encoder1.eval(), encoder2.eval(), encoder3.eval()
        for valid_loader in valid_loaders: 
            iterations = 0 
            class_loss_ptp = 0 
            class_loss_f1 = 0 
            with torch.no_grad(): 
                for batch in tqdm(valid_loader):
                    # initializatoins
                    optimizer.zero_grad()
                    images = batch['imgs'].cuda()
                    gt_samples = batch['samples'].cuda()
                    img_info = batch['img_info'].cuda()
                    obj_class = batch['class'][0]
                    batch_size  = images.shape[0]
                    initial_positions_batch = initial_positions.unsqueeze(0).expand(batch_size,num_verts, 3)

                    # extract image features
                    features1  = encoder1(images)
                    features2  = encoder2(images)
                    features3  = encoder3(images)

                    # loop for every image
                    # add batch normalization to models
                    vertex_features = batched_pooling(features1, initial_positions_batch, img_info.clone())
                    vertex_features, vertex_positions_1 = modelA(initial_positions_batch, vertex_features, adj_info['adj'])
                    vertex_positions_1 = initial_positions_batch + vertex_positions_1

                    vertex_features = torch.cat((vertex_features, batched_pooling(features2, vertex_positions_1.clone(), img_info.clone())), dim=-1)
                    vertex_features, vertex_positions_2 = modelB( vertex_positions_1.clone(), vertex_features, adj_info['adj'])
                    vertex_positions_2 = vertex_positions_2 + vertex_positions_1

                    vertex_features = torch.cat((vertex_features, batched_pooling(features3, vertex_positions_2.clone(), img_info.clone())), dim=-1 )
                    _,vertex_positions_3 = modelC(vertex_positions_2.clone(),vertex_features, adj_info['adj'])
                    vertex_positions_3 = vertex_positions_3 + vertex_positions_2    
              
                    # surface loss
                    loss, f1 = batch_point_to_point(vertex_positions_3, adj_info, gt_samples[:,:2466], num = 2466, f1 = True)
                    
                    
                    iterations += 1. 
                    class_loss_ptp += loss
                    class_loss_f1 += f1

                    print_loss = (class_loss_ptp / iterations)
                    print_loss_f1 = (class_loss_f1 / iterations)
                    
            message = f'Valid || Epoch: {self.epoch}, class: {obj_class}, f1: {print_loss_f1:.2f}, cur_f1: {self.best_f1:.2f}, ptp: {print_loss:.2f}, cur_ptp: {self.best_ptp:.2f}'
            tqdm.write(message)
                
            total_loss_ptp += ((class_loss_ptp / iterations) / 13.)
            total_loss_f1 += ((class_loss_f1 / iterations) /13.)

        print('*******************************************************')
        print(f'Total validation f1 is {total_loss_f1} and ptp is {total_loss_ptp}')
        print('*******************************************************')
        self.valid_loss_ptp.append((total_loss_ptp ))
        self.valid_loss_f1.append((total_loss_f1 ))

    def save(self): 
        if self.valid_loss_f1[-1] >= self.best_f1:
            improvement = - self.best_f1 + self.valid_loss_f1[-1] 
            print(f'Saving Mode with a {improvement} improvement in f1')
             
            torch.save(modelA.state_dict(), checkpoint_dir + 'model_1')
            torch.save(modelB.state_dict(), checkpoint_dir + 'model_2')
            torch.save(modelC.state_dict(), checkpoint_dir + 'model_3')
            torch.save(encoder1.state_dict(), checkpoint_dir + 'encoder_1')
            torch.save(encoder2.state_dict(), checkpoint_dir + 'encoder_2')
            torch.save(encoder3.state_dict(), checkpoint_dir + 'encoder_3')
            torch.save(optimizer.state_dict(), checkpoint_dir + 'optim')

            self.best_f1 = self.valid_loss_f1[-1]

        if self.valid_loss_ptp[-1] <= self.best_ptp:
            improvement = self.best_ptp - self.valid_loss_ptp[-1] 
            print(f'Got a {improvement} improvement in ptp')
            self.best_ptp = self.valid_loss_ptp[-1] 
    
    def graph(self): 
        plt.grid()
        plt.plot(self.train_loss, color='blue')
        plt.plot(self.valid_loss_ptp,color='red')
        plt.savefig( save_dir + 'ptp.png' )
        plt.clf()

        plt.grid()
        plt.plot(self.train_f1, color='blue')
        plt.plot(self.valid_loss_f1,color='red')
        plt.savefig( save_dir + 'f1.png' )
        plt.clf()

        print('***************************')
        print(f'Graphs have been created')
        print('**************************')

    def evaluate(self): 
        #load from old model 
        if args.pretrained:
            location = 'checkpoint/pretrained/'
        else: 
            location = f'checkpoint/{args.exp_id}/'
        modelA.load_state_dict(torch.load(location + 'model_1'))
        modelB.load_state_dict(torch.load(location + 'model_2'))
        modelC.load_state_dict(torch.load(location + 'model_3'))
        encoder1.load_state_dict(torch.load(location + 'encoder_1'))
        encoder2.load_state_dict(torch.load(location + 'encoder_2'))
        encoder3.load_state_dict(torch.load(location + 'encoder_3'))


        total_loss_f1 = 0 
        modelA.eval(), modelB.eval(), modelC.eval(), encoder1.eval(), encoder2.eval(), encoder3.eval()
        for valid_loader in valid_loaders: 
            iterations = 0 
            class_loss_f1 = 0 
            with torch.no_grad(): 
                for batch in tqdm(valid_loader):
                    
                    # initializatoins
                    optimizer.zero_grad()
                    images = batch['imgs'].cuda()
                    gt_samples = batch['samples'].cuda()
                    img_info = batch['img_info'].cuda()
                    obj_class = batch['class'][0]
                    batch_size  = images.shape[0]
                    initial_positions_batch = initial_positions.unsqueeze(0).expand(batch_size, num_verts, 3)

                    # extract image features
                    features1  = encoder1(images)
                    features2  = encoder2(images)
                    features3  = encoder3(images)

                    # first prediction    
                    vertex_features = batched_pooling(features1, initial_positions_batch, img_info.clone())
                    vertex_features, vertex_positions_1 = modelA(initial_positions_batch, vertex_features, adj_info['adj'])
                    vertex_positions_1 = initial_positions_batch + vertex_positions_1
                    
                    # second prediction
                    vertex_features = torch.cat((vertex_features, batched_pooling(features2, vertex_positions_1.clone(), img_info.clone())), dim=-1)
                    vertex_features, vertex_positions_2 = modelB( vertex_positions_1.clone(), vertex_features, adj_info['adj'])
                    vertex_positions_2 = vertex_positions_2 + vertex_positions_1
                    
                    # third prediction
                    vertex_features = torch.cat((vertex_features, batched_pooling(features3, vertex_positions_2.clone(), img_info.clone())), dim=-1 )
                    _,vertex_positions_3 = modelC(vertex_positions_2.clone(),vertex_features, adj_info['adj'])
                    vertex_positions_3 = vertex_positions_3 + vertex_positions_2    
              
                    if args.eval_vis: 
                        for image, verts in zip(images, vertex_positions_3): 
                            image = image.permute(1,2,0)[:,:,:3].data.cpu().numpy()
    
                            image = (image*255.).astype(np.uint8)
                            Image.fromarray(image).show()
                            render_mesh(verts.data.cpu().numpy(), adj_info['faces'].data.cpu().numpy())
                            message = f'Press any key for next model '
                            tqdm.write(message)
                            input()

                    # surface loss
                    _, f1 = batch_point_to_point(vertex_positions_3, adj_info, gt_samples[:,:2466], num = 2466, f1 = True)
                    
                    
                    iterations += 1. 
                    class_loss_f1 += f1
                    print_loss_f1 = (class_loss_f1 / iterations)
                    
            message = f'Valid || Epoch: {self.epoch}, class: {obj_class}, f1: {print_loss_f1 :.2f}'
            tqdm.write(message)
            total_loss_f1 += ((class_loss_f1 / iterations) /13.)


        print('*******************************************************')
        print(f'Total validation f1 is {total_loss_f1}')
        print('*******************************************************')




  

trainer = Engine()
if args.eval or args.eval_vis: 
    trainer.evaluate()
    exit()

for epoch in range(3000):

    trainer.epoch = epoch
    trainer.train()
    trainer.validate()
    trainer.save()
    if epoch> 0: 
        trainer.graph()
