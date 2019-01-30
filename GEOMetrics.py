from utils import *
from layers import *
from models import *
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--render', action='store_true', default=False,
                    help='render images from validation set')
parser.add_argument('--seed', type=int, default=41, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Initial learning rate.')
# parser.add_argument('--weight_decay', type=float, default=5e-6,
#                   help='Weight decay (L2 loss on parameters).')
parser.add_argument('--object', type=str, default='chair',
                    help='The object... ')

args = parser.parse_args()
batch_size = 5
loss_term = point_to_point


np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# establish directories
images = 'data/images/' + args.object +'/'
meshes = 'data/mesh_info/' +  args.object +'/'
samples = 'data/surfaces/' +  args.object +'/'
checkpoint_dir = "checkpoint/" +  args.object +'/'
save_dir =  "plots/" +  args.object +'/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# gather data
adj_info, initial_positions= load_initial()
initial_positions = Variable(initial_positions.cuda())
data_train = Mesh_loader(images + 'train/' ,meshes,samples)
data_valid = Mesh_loader(images + 'valid/' ,meshes, samples)


# initialize models
encoder1 = VGG()
encoder2 = VGG()
encoder3 = VGG()
modelA = MeshDeformationBlock(963)
modelB = MeshDeformationBlock(1155)
modelC = MeshDeformationBlock(1155)
encoder_mesh = MeshEncoder(50)
modelA.cuda(), encoder1.cuda(), encoder2.cuda(),encoder3.cuda(), modelB.cuda(), modelC.cuda(), encoder_mesh.cuda()
params = list(modelA.parameters()) + list(encoder1.parameters()) + list(encoder2.parameters()) + list(encoder3.parameters()) +list(modelB.parameters())  +list(modelC.parameters())
optimizer = optim.Adam(params,lr=args.lr)

try:
    encoder_mesh.load_state_dict(torch.load('checkpoint/'+ args.object + '/encoder'))
except:
    print 'You forgot to train the auto-encoder'
    exit()
if args.render:
    modelA.load_state_dict(torch.load(checkpoint_dir + 'model_1'))
    modelB.load_state_dict(torch.load(checkpoint_dir + 'model_2'))
    modelC.load_state_dict(torch.load(checkpoint_dir + 'model_3'))
    encoder1.load_state_dict(torch.load(checkpoint_dir + 'encoder_1'))
    encoder2.load_state_dict(torch.load(checkpoint_dir + 'encoder_2'))
    encoder3.load_state_dict(torch.load(checkpoint_dir + 'encoder_3'))



def train(loader, epoch, num_interations, iteration, best):
    # initializatoins
    t = time()
    edge_loss, lap_loss = 0,0
    modelA.train(), modelB.train(),modelC.train(), encoder1.train(), encoder2.train(), encoder3.train()
    optimizer.zero_grad()

    # load training batch
    batch = loader.load_batch(batch_size)


    # extract image features
    A1, B1, C1, D1  = encoder1(batch['imgs'])
    A2, B2, C2, D2  = encoder2(batch['imgs'])
    A3, B3, C3, D3  = encoder3(batch['imgs'])


    # initializations for graph convolutions
    full_info = zip( batch['verts'],batch['adjs'],batch['samples'], batch['latent'],batch['mats'], range(batch_size))
    loss = torch.FloatTensor(np.array(0.)).cuda()
    track_loss, track_lap, track_edge, track_lat, track_surface = 0, 0, 0, 0, 0
    real_distance = 0
    latent_loss = 0
    lap_loss = 0
    size = 0

    # loop for every image
    for gt,gt_adj,gt_samp,late, mat, i in full_info :

        # 3 mesh deformation layers
        # stage 1
        vertex_features = pooling([A1[i], B1[i], C1[i], D1[i]] , initial_positions, mat)
        vertex_features, vertex_positions_1 =modelA(initial_positions, vertex_features,adj_info)
        vertex_positions_1 = vertex_positions_1 + initial_positions
        splitting_faces = calc_curve(vertex_positions_1, adj_info)
        adj_info_2,vertex_positions_split_1,vertex_features= split_info(adj_info, vertex_positions_1, vertex_features, splitting_faces, number =splitting_faces.shape[0])

        # stage 2
        vertex_features = torch.cat((vertex_features, pooling([A2[i], B2[i], C2[i], D2[i]], vertex_positions_split_1, mat)), dim= 1 )
        vertex_features, vertex_positions_2 = modelB( vertex_positions_split_1, vertex_features, adj_info_2)
        vertex_positions_2 = vertex_positions_2 + vertex_positions_split_1
        splitting_faces = calc_curve(vertex_positions_2, adj_info_2)
        adj_info_3,vertex_positions_split_2,vertex_features= split_info(adj_info_2, vertex_positions_2, vertex_features, splitting_faces, number = splitting_faces.shape[0])

        # stage 3
        vertex_features = torch.cat((vertex_features, pooling([A3[i], B3[i], C3[i], D3[i]], vertex_positions_split_2, mat)), dim= 1 )
        _,vertex_positions_3 = modelC(vertex_positions_split_2,vertex_features, adj_info_3)
        vertex_positions_3 = vertex_positions_3 + vertex_positions_split_2

        size += vertex_positions_3.shape[0]



        ## latent loss, possible encoding of graph does not exist
        if late:
            copy_adj = adj_info_3.copy()
            latent_pred = encoder_mesh(vertex_positions_3, copy_adj )
            latent_gt   = encoder_mesh(gt, gt_adj )
            latent_loss = torch.mean(torch.abs(latent_pred - latent_gt)) * .001

        # surface loss
        surface_loss_1,_ = loss_term(vertex_positions_1, adj_info, gt_samp[:3000] ,num = 3000  )
        surface_loss_2,_ = loss_term(vertex_positions_2, adj_info_2, gt_samp[:3000],num = 3000  )
        surface_loss_3,_ = loss_term(vertex_positions_3, adj_info_3, gt_samp[:3000] ,num = 3000  )
        surface_loss = surface_loss_1 + surface_loss_2 + surface_loss_3
        real_distance += surface_loss_3.data.cpu().numpy()

        # edge length loss
        edge_loss = calc_edge(vertex_positions_1, adj_info) * 300
        edge_loss += calc_edge(vertex_positions_2, adj_info_2) * 300
        edge_loss += calc_edge(vertex_positions_3, adj_info_3) * 300
        edge_loss *= .6

        # laplacian loss
        lap_loss_1  = torch.mean(torch.sum((get_lap_info( initial_positions, adj_info) - get_lap_info( vertex_positions_1, adj_info))**2, 1 ))* 1500
        lap_loss_2  = torch.mean(torch.sum((get_lap_info( vertex_positions_split_1, adj_info_2) - get_lap_info( vertex_positions_2, adj_info_2))**2, 1)) * 1500
        lap_loss_2  += torch.mean(torch.sum((vertex_positions_split_1 - vertex_positions_2)**2, 1)) * 100
        lap_loss_3  = torch.mean(torch.sum((get_lap_info( vertex_positions_split_2, adj_info_3) - get_lap_info( vertex_positions_3, adj_info_3))**2, 1)) * 1500
        lap_loss_3  += torch.mean(torch.sum((vertex_positions_split_2- vertex_positions_3)**2, 1)) * 100
        lap_loss  = lap_loss_1*.3  + lap_loss_2  + lap_loss_3


        # combination of losses
        loss += edge_loss  +  latent_loss  + surface_loss + lap_loss
        # saving losses for printing
        track_edge += edge_loss / batch_size
        track_lap += lap_loss / batch_size
        track_lat += latent_loss / batch_size
        track_surface += surface_loss / batch_size
    loss /= batch_size

    loss.backward()
    optimizer.step()

    print 'Epoch: {:d}'.format(epoch),
    print ' Iter: {:d}/{:d}'.format(iteration, num_interations),
    print ' loss: {:.4f}'.format(track_loss),
    print ' surf_loss: {:.4f}'.format(track_surface),
    print ' latent: {:.4f}'.format(latent_loss),
    print ' e_loss: {:.4f}'.format(track_edge),
    print ' l_loss: {:.4f}'.format(track_lap),
    print ' size {:.4f}'.format(size/batch_size)
    print ' time: {:.2f}s'.format(time() - t),
    print ' best: {:.4f}'.format(best)
    return real_distance / batch_size

# calculate validation loss
def val( A1, B1, C1, D1 ,A2, B2, C2, D2 , A3, B3, C3, D3 ,   mat,samps, img , i, init  ):

    with torch.no_grad():
        vertex_positions = init
        vertex_features = pooling([A1, B1, C1, D1 ], init, mat)
        vertex_features, vertex_positions =modelA(init, vertex_features,adj_info)
        vertex_positions = vertex_positions + init
        splitting_faces = calc_curve(vertex_positions, adj_info)
        adj_info_2,vertex_positions, vertex_features= split_info(adj_info, vertex_positions, vertex_features, splitting_faces, number = splitting_faces.shape[0])
        vertex_features = torch.cat((vertex_features, pooling([A2, B2, C2, D2], vertex_positions, mat)), dim= 1 )
        vertex_features, vertex_positions_ = modelB(vertex_positions, vertex_features, adj_info_2)
        vertex_positions = vertex_positions + vertex_positions_
        splitting_faces = calc_curve(vertex_positions, adj_info_2)
        adj_info_3,vertex_positions, vertex_features= split_info(adj_info_2, vertex_positions, vertex_features, splitting_faces, number = splitting_faces.shape[0])
        vertex_features = torch.cat((vertex_features, pooling([A3, B3, C3, D3], vertex_positions, mat)), dim= 1 )
        _,vertex_positions_ = modelC(vertex_positions, vertex_features, adj_info_3)
        vertex_positions = vertex_positions + vertex_positions_
        loss, pred_points = point_to_point(vertex_positions, adj_info_3, samps[:2000] ,num = 2000  )
        pred_points.detach()
        loss.detach()
        optimizer.zero_grad()

        # to see output
        if args.render:
            active_faces_indecies = adj_info_3['face_list'][:,0,2].astype(int)
            all_faces = adj_info_3['faces']
            pred_face = all_faces[active_faces_indecies]
            pred_faces = torch.LongTensor(pred_face).cuda()
            render_mesh( vertex_positions.data.cpu().numpy(), pred_faces+1, img.permute(1,2,0).data.cpu().numpy())
            raw_input()

    return  float(loss.data.cpu().numpy())


def validate(batch):
    with torch.no_grad():
        optimizer.zero_grad()
        A1, B1, C1, D1 = encoder1(batch['imgs'])
        A2, B2, C2, D2 = encoder2(batch['imgs'])
        A3, B3, C3, D3 = encoder3(batch['imgs'])
        A1.detach() , B1.detach(), C1.detach(), D1.detach()
        A2.detach() , B2.detach(), C2.detach(), D2.detach()
        A3.detach() , B3.detach(), C3.detach(), D3.detach()
        full_info = zip( batch['mats'],batch['samples'], range(batch['mats'].shape[0]))
        loss = 0
        for  mat,samps,  i in full_info :
            cur_loss = val(A1[i].clone(), B1[i].clone(), C1[i].clone(), D1[i].clone(),
                A2[i].clone(), B2[i].clone(), C2[i].clone(), D2[i].clone(),
                A3[i].clone(), B3[i].clone(), C3[i].clone(), D3[i].clone()

                ,  mat,samps, batch['imgs'][i], i, initial_positions)
            torch.cuda.empty_cache()
            gc.collect()
            loss = loss +  cur_loss

        valid_loss = loss/ (i+1)


    return valid_loss


full_valid_loss, full_train_loss, length, min_val, real_val  = [],[], 50, 1000, 0
valid_batch = data_valid.load_batch(2, ordered = True)


if args.render:
    validate(valid_batch)
    exit()
for epoch in range(3000):

    if epoch == 1000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00003
    if epoch == 2000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00001
        loss_term = point_to_surface


    for i in range(length):
        full_train_loss.append(train(data_train, epoch, length, i, min_val))

    v_loss = validate(valid_batch)
    full_valid_loss.append(v_loss)

    if full_valid_loss[-1] < min_val:
        min_val = full_valid_loss[-1]
        torch.save(modelA.state_dict(), checkpoint_dir + 'model_1')
        torch.save(modelB.state_dict(), checkpoint_dir + 'model_2')
        torch.save(modelC.state_dict(), checkpoint_dir + 'model_3')
        torch.save(encoder1.state_dict(), checkpoint_dir + 'encoder_1')
        torch.save(encoder2.state_dict(), checkpoint_dir + 'encoder_2')
        torch.save(encoder3.state_dict(), checkpoint_dir + 'encoder_3')


    if epoch>5:
        graph(save_dir, full_train_loss[3:], full_valid_loss[3:])
