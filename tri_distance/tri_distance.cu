#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

__device__ 
struct Vec {
 	float x; 
 	float y; 
 	float z;
};

__device__ 
Vec make_vec(float x, float y, float z){
	Vec vec = {x, y, z}; 
	return vec;
};

__device__ Vec sub(Vec A, Vec B){
	Vec vec  = { A.x - B.x, A.y - B.y, A.z - B.z};
	return vec ;
}

__device__ Vec add(Vec A, Vec B){
	Vec vec  = { A.x + B.x, A.y + B.y, A.z + B.z};
	return vec ;
}

__device__  Vec mul(Vec A, float b){
	Vec vec  = { A.x*b, A.y*b, A.z*b};
	return vec;
}

__device__  Vec cross(Vec A, Vec B){
	Vec vec  = { A.y*B.z - A.z*B.y , A.z*B.x - A.x*B.z, A.x*B.y - A.y*B.x};
	return vec ;
}
__device__  float dot(Vec A, Vec B){ 
	return A.x*B.x + A.y*B.y + A.z*B.z;
}

__device__  float LengthSquared(Vec A){
	return dot(A,A);
}

__device__  float Project_Edge(Vec A, Vec D, Vec p ){
	Vec v = sub(p, A);
	float len = LengthSquared(D);
	float proj = dot(v, D) / len ;
	return proj ;

}

__device__  float vec_dist( Vec A, Vec B){
	Vec diff = sub(A, B);
	return LengthSquared(diff);
}

__device__  bool in_range(float a){
	if (a <=1 and a >=0){
		return true ;
	}
	else{
		return false;
	}
}
__device__  bool is_above(Vec A, Vec D, Vec T, Vec p ){
	Vec norm = cross(T, D);
	return dot(norm, sub(p ,A))> 0;

}

__device__  Vec point_at(Vec A, Vec D, float t){
	Vec v = add(A, mul(D, t));
	return v;
}
__device__ Vec normalize(Vec A){
	float len = sqrt(LengthSquared(A));
	A = mul(A, 1./len);
	return A;
}



__device__  Vec Project_Plane(Vec orig, Vec Direction, Vec point){
	Vec v = sub(point, orig);
	Vec unit_norm = normalize(Direction);
	float dist = dot(v, unit_norm);
	Vec projected_point = sub(point , mul(unit_norm, dist));
	return projected_point;
}


__global__ 
void TriDistanceKernel(
	int b,
	int n,
	const float* xyz,
	int m,
	const float* tri1,
    const float* tri2,
    const float* tri3,
	float* dist,
	int* point,
	int* index)
{
	const int batch=512;
	__shared__ float tri_buffA[batch*3];
	__shared__ float tri_buffB[batch*3];
	__shared__ float tri_buffC[batch*3];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int k2=0;k2<m;k2+=batch){
			int end_k=min(m,k2+batch)-k2;
			for (int j=threadIdx.x;j<end_k*3;j+=blockDim.x){
				tri_buffA[j]=tri1[(i*m+k2)*3+j];
				tri_buffB[j]=tri2[(i*m+k2)*3+j];
				tri_buffC[j]=tri3[(i*m+k2)*3+j];

			}
			__syncthreads();
			for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){


				Vec p1 = make_vec(xyz[(i*n+j)*3+0], xyz[(i*n+j)*3+1], xyz[(i*n+j)*3+2]);

				
				float best = 10000;
				float best_option=0;
				int end_ka=end_k-(end_k&3);
				Vec p2 = make_vec(1,1,1);
				int option = 0;
				int best_i = 0;
				
				for (int k=0;k<end_ka;k+=1){
					
						Vec A = make_vec(tri_buffA[k*3+0], tri_buffA[k*3+1], tri_buffA[k*3+2]);
						Vec B = make_vec(tri_buffB[k*3+0], tri_buffB[k*3+1], tri_buffB[k*3+2]);
						Vec C = make_vec(tri_buffC[k*3+0], tri_buffC[k*3+1], tri_buffC[k*3+2]);

						Vec AB_Delta = sub(B, A);
						Vec BC_Delta = sub(C, B);
						Vec CA_Delta = sub(A, C);

						Vec AB_A = A;
						Vec BC_A = B;
						Vec CA_A = C;

						Vec TriNorm = cross(sub(A,B), sub(A,C));

						float uab = Project_Edge(AB_A, AB_Delta, p1);
						float uca = Project_Edge(CA_A, CA_Delta, p1);

						if (uca > 1 && uab < 0){
							p2 = A;
							option = 1; 
						}
						else{ 
							float ubc = Project_Edge(BC_A, BC_Delta, p1);

							if (uab > 1 && ubc < 0){
            					p2 = B;
            					option = 2;

							}
							else if (ubc > 1 && uca < 0){
								p2 = C;
								option = 3;
							}

							else{
								if (in_range(uab) &&  (not is_above(AB_A, AB_Delta, TriNorm, p1))){
									p2 = point_at( AB_A, AB_Delta, uab);
									option = 4;
								}
								else if (in_range(ubc) &&  (not is_above(BC_A, BC_Delta, TriNorm, p1))){
									p2 = point_at( BC_A, BC_Delta, ubc);
									option = 5;
								}
								else if (in_range(uca) &&  (not is_above(CA_A, CA_Delta, TriNorm, p1))){
									p2 = point_at( CA_A, AB_Delta, uca);
									option = 6;
								}
								else{
									p2 = Project_Plane(A, TriNorm, p1);
									option = 0; 
									
								}
							}
						}

						float d = vec_dist( p1, p2);

	
						if (k==0 || d<best){
							best=d;
							best_option = option;
							best_i=k+k2;
						}
					
				}

				if (k2==0 || dist[(i*n+j)]>best){
					dist[(i*n+j)]=best;
					point[(i*n+j)]=best_option;
					index[(i*n+j)]=best_i;
				}
			}
			__syncthreads();
		}
	}
}

void TriDistanceKernelLauncher(const int b, const int n,
    const float* xyz,
    const int m,
    const float* tri1,
    const float* tri2,
    const float* tri3,
    float* dist,
    int* point,
    int* index)
{
	TriDistanceKernel<<<dim3(32,16,1),512>>>(b, n, xyz, m, tri1, tri2, tri3, dist, point, index);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	    printf("error in chamfer distance updateOutput: %s\n", cudaGetErrorString(err));
}


