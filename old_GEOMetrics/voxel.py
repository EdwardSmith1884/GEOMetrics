import numpy as np
import os 
def call(command):
    os.system('%s > /dev/null 2>&1' % command)

def voxel2obj(filename, pred, show='False', threshold=0.4):
    verts, faces = voxel2mesher(pred, threshold )
    write_obj(filename, verts, faces)
    call('meshlab ' + filename)



def write_obj(filename, verts, faces):
    """ write the verts and faces on file."""
    with open(filename, 'w') as f:
        # write vertices
        f.write('g\n# %d vertex\n' % len(verts))
        for vert in verts:
            f.write('v %f %f %f\n' % tuple(vert))

        # write faces
        f.write('# %d faces\n' % len(faces))
        for face in faces:
            f.write('f %d %d %d\n' % tuple(face))





def mesh2obj( verts, faces):
    filename = 'temp.obj'
    write_obj(filename, verts, faces)
    call('meshlab ' + filename)




