#include <algorithm>
#include <cmath>
#include <vector>

#include <chrono>
#include <omp.h>
#include <thread>

using namespace std;

/*
 *==============================================================================
 * If you want to compile this file for verification purposes, use this option:
 *
 *         g++ -c -std=c++11 -fopenmp miscellaneous_implementation.cpp
 *==============================================================================
 */

void test_parallelism(int iterations) {
	#pragma omp parallel

	#pragma omp for
	for(int i = 0; i < iterations; i++) {
		this_thread::sleep_for(chrono::nanoseconds(1000000000));
	}
}

void atomic_ind_dip(double * coors, double * charges, double * alpha, double * coors_ind, double * prev_atomic_dipoles, double * atomic_dipoles, int n_particles_all, int n_particles_ind, double boxlength) {
	/*
	 * Shapes:
	 * =======
	 * coors               ... (N_all, 3)
	 * charges             ... (N_all)
	 * alpha               ... (N_ind)
	 * coors_ind           ... (N_ind, 3)
	 * atomic_dipoles      ... (N_ind, 3)
	 * prev_atomic_dipoles ... (N_ind, 3)
	 */
	int atom1, atom2, i,j,k, dim;
	double dist, dist_sq, dist_cub;
	double dist_vec[3];
	double dip_ten[3][3];
	double e_0[n_particles_ind*3]; // Shape ... (N_ind, 3)
	double boxlength_2 = boxlength / 2.0;

	for(i = 0; i < n_particles_ind*3; i++) {
		e_0[i] = 0;
	}

	for(atom1 = 0; atom1 < n_particles_ind; atom1++) {
		for(atom2 = 0; atom2 < n_particles_all; atom2++) {
			dist_sq = 0;
			for(dim = 0; dim < 3; dim++) {
				dist = coors[atom2*3+dim] - coors[atom1*3+dim];
				dist_sq += pow(dist, 2);
			}
			if(dist_sq == 0) {continue;}

			dist_cub = pow(dist_sq, (-1.5));
			for(dim = 0; dim < 3; dim++) {
				e_0[atom1*3+dim] += charges[atom1] * charges[atom2] * (coors[atom2*3+dim] - coors[atom1*3+dim]) * dist_cub;
			}
		}
	}

	if((atomic_dipoles[0] == 0) and (atomic_dipoles[1] == 0) and (atomic_dipoles[2] == 0)) {
		for(atom1 = 0; atom1 < n_particles_ind; atom1++) {
			for(dim = 0; dim < 3; dim++) {
				prev_atomic_dipoles[atom1*3+dim] = e_0[atom1*3+dim];
			}
		}
	}

	for(i = 0; i < 4; i++) {
		for(atom1 = 0; atom1 < n_particles_ind; atom1++) {
			for(dim = 0; dim < 3; dim++) {
				atomic_dipoles[atom1*3+dim] = alpha[atom1] * e_0[atom1*3+dim];
			}

			for(atom2 = 0; atom2 < n_particles_ind; atom2++) {
				if(atom1 == atom2) continue;

				dist_sq = 0;
				for(dim = 0; dim < 3; dim++) {
					dist_vec[dim] = coors_ind[atom2*3+dim] - coors_ind[atom1*3+dim];
					dist_sq += pow(dist_vec[dim],2);
				}
				dist_cub = pow(dist_sq, -1.5);

				for(j = 0; j < 3; j++) {
					for(k = 0; k < 3; k++) {
						dip_ten[j][k] = dist_vec[j] * dist_vec[k] / dist_sq;
						if(j == k) {dip_ten[j][k] -= 1.0;}
						dip_ten[j][k] *= dist_cub;
					}
				}

				for(dim = 0; dim < 3; dim++) {
					atomic_dipoles[atom1*3+dim] += alpha[atom1] * (dip_ten[dim][0]*prev_atomic_dipoles[atom2*3+dim] + dip_ten[dim][1]*prev_atomic_dipoles[atom2*3+dim] + dip_ten[dim][2]*prev_atomic_dipoles[atom2*3+dim]);
				}
			}
		}

		for(atom1 = 0; atom1 < n_particles_ind; atom1++) {
			for(dim = 0; dim < 3; dim++) {
				prev_atomic_dipoles[atom1*3+dim] = atomic_dipoles[atom1*3+dim];
			}
		}
	}
}

void atomic_ind_dip_per_atom(int atom1, double * coors, double * charges, double * alpha, double * coors_ind, double * prev_atomic_dipoles, double * atomic_dipoles, int n_particles_all, int n_particles_ind, double boxlength) {
	/*
	 * Shapes:
	 * =======
	 * coors               ... (N_all, 3)
	 * charges             ... (N_all)
	 * alpha               ... (N_ind)
	 * coors_ind           ... (N_ind, 3)
	 * atomic_dipoles      ... (N_ind, 3)
	 * prev_atomic_dipoles ... (N_ind, 3)
	 */
	int atom2, i,j,k, dim;
	double dist, dist_sq, dist_cub;
	double dist_vec[3];
	double dip_ten[3][3];
	double e_0[3];
	double boxlength_2 = boxlength / 2.0;

	//double e_0_x = 0, e_0_y = 0, e_0_z = 0;

	for(dim = 0; dim < 3; dim++) {
		e_0[dim] = 0;
	}

	//# pragma omp parallel default(none) shared(coors, charges, alpha, coors_ind, prev_atomic_dipoles, n_particles_all, n_particles_ind, boxlength) private(dim, dist, dist_sq, dist_cub, atom1, atom2) reduction(+:e_0_x, e_0_y, e_0_z)
	{
		//#pragma omp for
		for(atom2 = 0; atom2 < n_particles_all; atom2++) {
			dist_sq = 0;
			for(dim = 0; dim < 3; dim++) {
				dist = coors[atom2*3+dim] - coors[atom1*3+dim];
				dist_sq += pow(dist, 2);
			}
			if(dist_sq == 0) {continue;}

			dist_cub = pow(dist_sq, (-1.5));
			e_0[0] += charges[atom1] * charges[atom2] * (coors[atom2*3+0] - coors_ind[atom1*3+0]) * dist_cub; //Last coors: coors or coors_ind?
			e_0[1] += charges[atom1] * charges[atom2] * (coors[atom2*3+1] - coors_ind[atom1*3+1]) * dist_cub; //Last coors: coors or coors_ind?
			e_0[2] += charges[atom1] * charges[atom2] * (coors[atom2*3+2] - coors_ind[atom1*3+2]) * dist_cub; //Last coors: coors or coors_ind?
			//e_0_x += charges[atom1] * charges[atom2] * (coors[atom2*3+0] - coors_ind[atom1*3+0]) * dist_cub; //Last coors: coors or coors_ind?
			//e_0_y += charges[atom1] * charges[atom2] * (coors[atom2*3+1] - coors_ind[atom1*3+1]) * dist_cub; //Last coors: coors or coors_ind?
			//e_0_z += charges[atom1] * charges[atom2] * (coors[atom2*3+2] - coors_ind[atom1*3+2]) * dist_cub; //Last coors: coors or coors_ind?
		}
	}
	//e_0[0] = e_0_x;
	//e_0[1] = e_0_y;
	//e_0[2] = e_0_z;

	if((atomic_dipoles[0] == 0) and (atomic_dipoles[1] == 0) and (atomic_dipoles[2] == 0)) {
		for(dim = 0; dim < 3; dim++) {
			prev_atomic_dipoles[atom1*3+dim] = e_0[dim];
		}
	}

	for(i = 0; i < 4; i++) {
		for(dim = 0; dim < 3; dim++) {
			atomic_dipoles[atom1*3+dim] = alpha[atom1] * e_0[dim];
		}

		//double atomic_dipoles_x = 0, atomic_dipoles_y = 0, atomic_dipoles_z = 0;

		//#pragma omp parallel default(none) shared(n_particles_ind, coors_ind, alpha, prev_atomic_dipoles) private(dip_ten, dim, j, k, atom1, atom2, dist_sq, dist_cub, dist_vec) reduction(+:atomic_dipoles_x,atomic_dipoles_y,atomic_dipoles_z)
		{
			//#pragma omp for
			for(atom2 = 0; atom2 < n_particles_ind; atom2++) {
				if(atom1 == atom2) continue;

				dist_sq = 0;
				for(dim = 0; dim < 3; dim++) {
					dist_vec[dim] = coors_ind[atom2*3+dim] - coors_ind[atom1*3+dim];
					dist_sq += pow(dist_vec[dim],2);
				}
				dist_cub = pow(dist_sq, -1.5);

				for(j = 0; j < 3; j++) {
					for(k = 0; k < 3; k++) {
						dip_ten[j][k] = 3 * dist_vec[j] * dist_vec[k] / dist_sq;
						if(j == k) {dip_ten[j][k] -= 1.0;}
						dip_ten[j][k] *= dist_cub;
					}
				}

				atomic_dipoles[atom1*3+0] += alpha[atom1] * (dip_ten[0][0]*prev_atomic_dipoles[atom2*3+0] + dip_ten[0][1]*prev_atomic_dipoles[atom2*3+0] + dip_ten[0][2]*prev_atomic_dipoles[atom2*3+0]);
				atomic_dipoles[atom1*3+1] += alpha[atom1] * (dip_ten[1][0]*prev_atomic_dipoles[atom2*3+1] + dip_ten[1][1]*prev_atomic_dipoles[atom2*3+1] + dip_ten[1][2]*prev_atomic_dipoles[atom2*3+1]);
				atomic_dipoles[atom1*3+2] += alpha[atom1] * (dip_ten[2][0]*prev_atomic_dipoles[atom2*3+2] + dip_ten[2][1]*prev_atomic_dipoles[atom2*3+2] + dip_ten[2][2]*prev_atomic_dipoles[atom2*3+2]);
				//atomic_dipoles_x += alpha[atom1] * (dip_ten[0][0]*prev_atomic_dipoles[atom2*3+0] + dip_ten[0][1]*prev_atomic_dipoles[atom2*3+0] + dip_ten[0][2]*prev_atomic_dipoles[atom2*3+0]);
				//atomic_dipoles_y += alpha[atom1] * (dip_ten[1][0]*prev_atomic_dipoles[atom2*3+1] + dip_ten[1][1]*prev_atomic_dipoles[atom2*3+1] + dip_ten[1][2]*prev_atomic_dipoles[atom2*3+1]);
				//atomic_dipoles_z += alpha[atom1] * (dip_ten[2][0]*prev_atomic_dipoles[atom2*3+2] + dip_ten[2][1]*prev_atomic_dipoles[atom2*3+2] + dip_ten[2][2]*prev_atomic_dipoles[atom2*3+2]);
			}
		}

		//atomic_dipoles[atom1*3+0] = atomic_dipoles_x;
		//atomic_dipoles[atom1*3+1] = atomic_dipoles_y;
		//atomic_dipoles[atom1*3+2] = atomic_dipoles_z;

		for(dim = 0; dim < 3; dim++) {
			prev_atomic_dipoles[atom1*3+dim] = atomic_dipoles[atom1*3+dim];
		}
	}
}

inline double kronecker(int a, int b) {
	if (a == b) return 1.0;
	else return 0.0;
}

void derive_ind_dip(double * coors_ind, double * vel_ind, double * atomic_dipoles, double * derived_atomic_dipoles, int n_particles_ind, double boxlength) {
	/*
	 * Shapes:
	 * =======
	 * coors_ind              ... (n_particles_ind, 3)
	 * vel_ind                ... (n_particles_ind, 3)
	 * atomic_dipoles         ... (n_particles_ind, 3)
	 * derived_atomic_dipoles ... (n_particles_ind, 3)
	 */

	int dim1, dim2, dim3;
	double dip_ten[3][3][3];
	int atom1, atom2;
	double dist_sq, dist_5, dist_7;
	double dist_vec[3];
	double t_mu[3];
	double boxlength_2 = boxlength / 2.0;

	for(atom1 = 0; atom1 < n_particles_ind; atom1++) {
		for(atom2 = 0; atom2 < n_particles_ind; atom2++) {
			if(atom1 == atom2) continue;

			for(dim1 = 0; dim1 < 3; dim1++) {
				dist_vec[dim1] = coors_ind[atom2*3+dim1] - coors_ind[atom1*3+dim1];
			}
			dist_sq = pow(dist_vec[0],2) + pow(dist_vec[1],2) + pow(dist_vec[2],2);
			dist_5  = pow(dist_sq, (-2.5));
			dist_7  = pow(dist_sq, (-3.5));

			for(dim1 = 0; dim1 < 3; dim1++) {
				for(dim2 = 0; dim2 < 3; dim2++) {
					for(dim3 = 0; dim3 < 3; dim3++) {
						dip_ten[dim1][dim2][dim3] = 3*dist_5*(kronecker(dim2,dim3)*dist_vec[dim1] + kronecker(dim3,dim1)*dist_vec[dim2] + kronecker(dim1,dim2)*dist_vec[dim3]) - 15*dist_7*dist_vec[dim1]*dist_vec[dim2]*dist_vec[dim3];
					}
				}
			}

			for(dim3 = 0; dim3 < 3; dim3++) {
				for(dim2 = 0; dim2 < 3; dim2++) {
					t_mu[dim2] = dip_ten[0][dim2][dim3] * atomic_dipoles[atom1*3+dim2] + dip_ten[1][dim2][dim3] * atomic_dipoles[atom1*3+dim2] + dip_ten[2][dim2][dim3] * atomic_dipoles[atom1*3+dim2];
				}

				for(dim1 = 0; dim1 < 3; dim1++) {
					derived_atomic_dipoles[atom1*3+dim3] += vel_ind[atom1*3 + dim1] * t_mu[dim1];
				}
			}
		}
	}
}


void derive_ind_dip_per_atom(int atom1, double * coors_ind, double * vel_ind, double * atomic_dipoles, double * derived_atomic_dipoles, int n_particles_ind, double boxlength) {
	/*
	 * Shapes:
	 * =======
	 * coors_ind              ... (n_particles_ind, 3)
	 * vel_ind                ... (n_particles_ind, 3)
	 * atomic_dipoles         ... (n_particles_ind, 3)
	 * derived_atomic_dipoles ... (n_particles_ind, 3)
	 */

	int dim1, dim2, dim3;
	double dip_ten[3][3][3];
	int atom2;
	double dist_sq, dist_5, dist_7;
	double dist_vec[3];
	double t_mu[3];
	double boxlength_2 = boxlength / 2.0;

	//double derived_atomic_dipoles_x = 0, derived_atomic_dipoles_y = 0, derived_atomic_dipoles_z = 0;

	//#pragma omp parallel default(none) shared(boxlength, boxlength_2, atom1, coors_ind, vel_ind, atomic_dipoles, n_particles_ind) private(dim1, dim2, dim3, dip_ten, atom2, dist_sq, dist_5, dist_7, dist_vec, t_mu) reduction(+:derived_atomic_dipoles_x, derived_atomic_dipoles_y, derived_atomic_dipoles_z)
	{
		//# pragma omp for
		for(atom2 = 0; atom2 < n_particles_ind; atom2++) {
			if(atom1 == atom2) continue;

			for(dim1 = 0; dim1 < 3; dim1++) {
				dist_vec[dim1] = coors_ind[atom2*3+dim1] - coors_ind[atom1*3+dim1];
			}
			dist_sq = pow(dist_vec[0],2) + pow(dist_vec[1],2) + pow(dist_vec[2],2);
			dist_5  = pow(dist_sq, (-2.5));
			dist_7  = pow(dist_sq, (-3.5));

			for(dim1 = 0; dim1 < 3; dim1++) {
				for(dim2 = 0; dim2 < 3; dim2++) {
					for(dim3 = 0; dim3 < 3; dim3++) {
						dip_ten[dim1][dim2][dim3] = 3*dist_5*(kronecker(dim2,dim3)*dist_vec[dim1] + kronecker(dim3,dim1)*dist_vec[dim2] + kronecker(dim1,dim2)*dist_vec[dim3]) - 15*dist_7*dist_vec[dim1]*dist_vec[dim2]*dist_vec[dim3];
					}
				}
			}

			for(dim3 = 0; dim3 < 3; dim3++) {
				for(dim2 = 0; dim2 < 3; dim2++) {
					t_mu[dim2] = dip_ten[0][dim2][dim3] * atomic_dipoles[atom1*3+dim2] + dip_ten[1][dim2][dim3] * atomic_dipoles[atom1*3+dim2] + dip_ten[2][dim2][dim3] * atomic_dipoles[atom1*3+dim2];
				}

				for(dim1 = 0; dim1 < 3; dim1++) {
					derived_atomic_dipoles[atom1*3+dim3] = vel_ind[atom1*3 + dim1] * t_mu[dim1];
					derived_atomic_dipoles[atom1*3+dim3] = vel_ind[atom1*3 + dim1] * t_mu[dim1];
					derived_atomic_dipoles[atom1*3+dim3] = vel_ind[atom1*3 + dim1] * t_mu[dim1];
					//derived_atomic_dipoles_x += vel_ind[atom1*3 + dim1] * t_mu[dim1];
					//derived_atomic_dipoles_y += vel_ind[atom1*3 + dim1] * t_mu[dim1];
					//derived_atomic_dipoles_z += vel_ind[atom1*3 + dim1] * t_mu[dim1];
				}
			}
		}
	}

	//derived_atomic_dipoles[atom1*3+0] = derived_atomic_dipoles_x;
	//derived_atomic_dipoles[atom1*3+1] = derived_atomic_dipoles_y;
	//derived_atomic_dipoles[atom1*3+2] = derived_atomic_dipoles_z;
}


void calc_accumulate_shellwise_g_function(double * aufpunkte, double * coor, char * dataset, double * histogram, double * norm, int n_aufpunkte, int n_particles, int number_of_datapoints, int maxshell, double histo_min, double histo_max, double boxlength) {
	int i, j;
	int shell, bin;
	double dist_x_sq, dist_y_sq, dist_z_sq;
	double dist, dist_sq, shortest_dist_sq;
	double histo_dx = (histo_max - histo_min) / double(number_of_datapoints - 1);
	double boxlength_sq = pow(boxlength, 2);
	double cutoff = pow(boxlength, 2)/4;

	for(i = 0; i < n_particles; i++) {
		shell = (int) dataset[i] - 1;

		(shell < maxshell) ? (shell = shell) : (shell = maxshell);

		shortest_dist_sq = pow((coor[3*i+0]-aufpunkte[3*0+0]), 2) +  pow((coor[3*i+1]-aufpunkte[3*0+1]), 2) +  pow((coor[3*i+2]-aufpunkte[3*0+2]), 2);

		for (j = 1; j < n_aufpunkte; j++) {
			dist_x_sq = pow((coor[3*i+0]-aufpunkte[3*j+0]), 2);
			while(dist_x_sq > cutoff) {
				dist_x_sq -= boxlength_sq;
			}

			dist_y_sq = pow((coor[3*i+1]-aufpunkte[3*j+1]), 2);
			while(dist_y_sq > cutoff) {
				dist_y_sq -= boxlength_sq;
			}

			dist_z_sq = pow((coor[3*i+2]-aufpunkte[3*j+2]), 2);
			while(dist_z_sq > cutoff) {
				dist_z_sq -= boxlength_sq;
			}

			dist_x_sq = abs(dist_x_sq);
			dist_y_sq = abs(dist_y_sq);
			dist_z_sq = abs(dist_z_sq);
			dist_sq = dist_x_sq + dist_y_sq + dist_z_sq;

			if(dist_sq < shortest_dist_sq) {
				shortest_dist_sq = dist_sq;
			}
		}

		dist = sqrt(shortest_dist_sq);

		if((dist > histo_min) && (dist < histo_max)) {
			bin = int( (dist - histo_min + histo_dx*0.5) /histo_dx );

			norm[shell] += 1;
			histogram[shell*number_of_datapoints + bin] += 1;
		}
	}
}

/*
void calc_accumulate_shellwise_g_function(double * aufpunkte, double * coor, char * dataset, double * histogram, double * norm, int n_aufpunkte, int n_particles, int number_of_datapoints, int maxshell, double histo_min, double histo_max, double boxlength) {
	int i, j;
	int shell, bin;
	double dist_x_sq, dist_y_sq, dist_z_sq;
	double dist, dist_sq, shortest_dist_sq;
	double histo_dx = (histo_max - histo_min) / double(number_of_datapoints - 1);
	double boxlength_sq = pow(boxlength, 2);
	double cutoff = pow(boxlength, 2)/4;

	for(i = 0; i < n_particles; i++) {
		shell = (int) dataset[i] - 1;

		(shell < maxshell) ? (shell = shell) : (shell = maxshell);

/		shortest_dist_sq = pow((coor[3*i+0]-aufpunkte[3*0+0]), 2) +  pow((coor[3*i+1]-aufpunkte[3*0+1]), 2) +  pow((coor[3*i+2]-aufpunkte[3*0+2]), 2);

		for (j = 0; j < n_aufpunkte; j++) {
			//dist_sq = pow((coor[3*i+0]-aufpunkte[3*j+0]), 2) +  pow((coor[3*i+1]-aufpunkte[3*j+1]), 2) +  pow((coor[3*i+2]-aufpunkte[3*j+2]), 2);

			dist_x_sq = pow((coor[3*i+0]-aufpunkte[3*j+0]), 2);
			while(dist_x_sq > cutoff) {
				dist_x_sq -= boxlength_sq;
			}

			dist_y_sq = pow((coor[3*i+1]-aufpunkte[3*j+1]), 2);
			while(dist_y_sq > cutoff) {
				dist_y_sq -= boxlength_sq;
			}

			dist_z_sq = pow((coor[3*i+2]-aufpunkte[3*j+2]), 2);
			while(dist_z_sq > cutoff) {
				dist_z_sq -= boxlength_sq;
			}

			dist_x_sq = abs(dist_x_sq);
			dist_y_sq = abs(dist_y_sq);
			dist_z_sq = abs(dist_z_sq);
			dist_sq = dist_x_sq + dist_y_sq + dist_z_sq;
/
/			if(dist_sq < shortest_dist_sq) {
/				shortest_dist_sq = dist_sq;
/			}
/		}
/
			dist = sqrt(dist_sq);

			if((dist > histo_min) && (dist < histo_max)) {
				bin = int( (dist - histo_min + histo_dx*0.5) /histo_dx );

				norm[shell] += 1;
				histogram[shell*number_of_datapoints + bin] += 1/float(n_aufpunkte);
			}
		}
	}
}
*/

int get_best_index(double * aufpunkt, double * coor, int n_particles) {
	int i = 0, best_index = 0;
	double curr_dist_sq, best_dist_sq = (coor[0] - aufpunkt[0])*(coor[0] - aufpunkt[0]) + (coor[1] - aufpunkt[1])*(coor[1] - aufpunkt[1]) + (coor[2] - aufpunkt[2])*(coor[2] - aufpunkt[2]);

	for(i = 1; i < n_particles; i++) {
		curr_dist_sq = (coor[3*i] - aufpunkt[0])*(coor[3*i] - aufpunkt[0]) + (coor[3*i+1] - aufpunkt[1])*(coor[3*i+1] - aufpunkt[1]) + (coor[3*i+2] - aufpunkt[2])*(coor[3*i+2] - aufpunkt[2]);
		if (curr_dist_sq < best_dist_sq) {best_index = i; best_dist_sq = curr_dist_sq;}
	}

	return best_index;
}

void write_Mr_diagram(double * coor, double * dip, int n_particles, double * aufpunkt, double * antagonist, double * histogram, double max_distance, int segments_per_angstroem, int order = 0) {
	/*
	   ATTN: This function merely accumulates and divides through an entry counter; normalization of dipoles,... must be done in the main applying program!
	*/
	int i, dist_bin;
	double dist_sq;
	vector<double> norm(int(max_distance*segments_per_angstroem)+1);
	for(i = 0; i < int(max_distance*segments_per_angstroem)+1; i++) {
		norm.at(i) = 0;
	}

	for(i = 0; i < n_particles; i++) {
		dist_sq = pow((coor[3*i] - aufpunkt[0]), 2) + pow((coor[3*i+1] - aufpunkt[1]), 2) + pow((coor[3*i+2] - aufpunkt[2]), 2);

		if(dist_sq <= max_distance*max_distance) {
			dist_bin = int(sqrt(dist_sq)*segments_per_angstroem);
			if(dist_bin >= 0 && dist_bin < int(max_distance*segments_per_angstroem)) {
				if(order == 0) {
					histogram[dist_bin] += dip[3*i]*antagonist[0] + dip[3*i+1]*antagonist[1] + dip[3*i+2]*antagonist[2];
					norm.at(dist_bin) += 1;
				}
				//TODO: Implement Legendrian orders
			}
		}
	}

	for(i = 0; i < int(max_distance*segments_per_angstroem); i++) {
		if(norm.at(i) > 1) {
			histogram[i] /= norm.at(i);
		}
	}
}

void write_Kirkwood_diagram(double * coor_aufpunkt, double * dip_aufpunkt, double * coor, double * dip, int n_particles, double * histogram, double max_distance, int segments_per_angstroem, int order = 1) {
	int i, dist_bin;
	double cos_theta, norm = dip_aufpunkt[0]*dip_aufpunkt[0] + dip_aufpunkt[1]*dip_aufpunkt[1] + dip_aufpunkt[2]*dip_aufpunkt[2];

	for(i = 0; i < n_particles; i++) {
		dist_bin = int(segments_per_angstroem * sqrt((coor[3*i] - coor_aufpunkt[0])*(coor[3*i] - coor_aufpunkt[0]) + (coor[3*i+1] - coor_aufpunkt[1])*(coor[3*i+1] - coor_aufpunkt[1]) + (coor[3*i+2] - coor_aufpunkt[2])*(coor[3*i+2] - coor_aufpunkt[2])));

		cos_theta = (( (dip_aufpunkt[0]*dip[3*i]) + (dip_aufpunkt[1]*dip[3*i+1]) + (dip_aufpunkt[2]*dip[3*i+2]) )/norm);

		if (dist_bin < max_distance * segments_per_angstroem) { 
			if(order == 1) {histogram[dist_bin] += cos_theta;}
			if(order == 2) {histogram[dist_bin] += (1.5 * pow(cos_theta,2) - 0.5);}
			if(order == 3) {histogram[dist_bin] += (2.5 * pow(cos_theta, 3) - 1.5 * cos_theta);}
			if(order == 4) {histogram[dist_bin] += (4.375 * pow(cos_theta, 4) - 3.75 * pow(cos_theta, 2) + 0.375);}
			if(order == 5) {histogram[dist_bin] += (7.875 * pow(cos_theta, 5) - 8.75 * pow(cos_theta, 3) + 1.875 * cos_theta);}
			if(order == 6) {histogram[dist_bin] += (14.4375 * pow(cos_theta, 6) - 19.6875 * pow(cos_theta, 4) + 6.5625 * pow(cos_theta, 2) - 0.3145);}
		}
	}
}

void write_Kirkwood_diagram_shellwise(double * coor_aufpunkt, double * dip_aufpunkt, double * coor, double * dip, int n_particles, double * histogram, char * dataset, int maxshell, double max_distance, int segments_per_angstroem, int order = 1) {
	int i, shell, dist_bin;

	double cos_theta, norm = dip_aufpunkt[0]*dip_aufpunkt[0] + dip_aufpunkt[1]*dip_aufpunkt[1] + dip_aufpunkt[2]*dip_aufpunkt[2];

	for(i = 0; i < n_particles; i++) {
		shell = (int) dataset[i] - 1;
		(shell <= maxshell) ? (shell = shell) : (shell = maxshell-1);

		dist_bin = int(segments_per_angstroem * sqrt((coor[3*i] - coor_aufpunkt[0])*(coor[3*i] - coor_aufpunkt[0]) + (coor[3*i+1] - coor_aufpunkt[1])*(coor[3*i+1] - coor_aufpunkt[1]) + (coor[3*i+2] - coor_aufpunkt[2])*(coor[3*i+2] - coor_aufpunkt[2])));

		cos_theta = (( (dip_aufpunkt[0]*dip[3*i]) + (dip_aufpunkt[1]*dip[3*i+1]) + (dip_aufpunkt[2]*dip[3*i+2]) )/norm);

		if (dist_bin < max_distance * segments_per_angstroem) { 
			if(order == 1) {histogram[shell*int(max_distance*segments_per_angstroem) + dist_bin] += cos_theta;}
			if(order == 2) {histogram[shell*int(max_distance*segments_per_angstroem) + dist_bin] += (1.5 * pow(cos_theta,2) - 0.5);}
			if(order == 3) {histogram[shell*int(max_distance*segments_per_angstroem) + dist_bin] += (2.5 * pow(cos_theta, 3) - 1.5 * cos_theta);}
			if(order == 4) {histogram[shell*int(max_distance*segments_per_angstroem) + dist_bin] += (4.375 * pow(cos_theta, 4) - 3.75 * pow(cos_theta, 2) + 0.375);}
			if(order == 5) {histogram[shell*int(max_distance*segments_per_angstroem) + dist_bin] += (7.875 * pow(cos_theta, 5) - 8.75 * pow(cos_theta, 3) + 1.875 * cos_theta);}
			if(order == 6) {histogram[shell*int(max_distance*segments_per_angstroem) + dist_bin] += (14.4375 * pow(cos_theta, 6) - 19.6875 * pow(cos_theta, 4) + 6.5625 * pow(cos_theta, 2) - 0.3145);}
		}
	}
}

void write_Kirkwood_diagram_2D(double * coor_aufpunkt, double * dip_aufpunkt, double * coor, double * dip, int n_particles, double * histogram, double max_distance, int segments_per_angstroem, int segments_per_pi, int order = 1) {
	int i, dist_bin, cos_bin, entry;
	double dipolar_cos, dipolar_cos_pos;
	double norm = dip_aufpunkt[0]*dip_aufpunkt[0] + dip_aufpunkt[1]*dip_aufpunkt[1] + dip_aufpunkt[2]*dip_aufpunkt[2];

	for(i = 0; i < n_particles; i++) {
		dist_bin = int(segments_per_angstroem * sqrt((coor[3*i] - coor_aufpunkt[0])*(coor[3*i] - coor_aufpunkt[0]) + (coor[3*i+1] - coor_aufpunkt[1])*(coor[3*i+1] - coor_aufpunkt[1]) + (coor[3*i+2] - coor_aufpunkt[2])*(coor[3*i+2] - coor_aufpunkt[2])));

		dipolar_cos = (dip[3*i]*dip_aufpunkt[0] + dip[3*i+1]*dip_aufpunkt[1] + dip[3*i+2]*dip_aufpunkt[2])/norm;

		if(order == 1) {dipolar_cos_pos = dipolar_cos;}
		if(order == 2) {dipolar_cos_pos = (1.5 * pow(dipolar_cos,2) - 0.5);}
		if(order == 3) {dipolar_cos_pos = (2.5 * pow(dipolar_cos, 3) - 1.5 * dipolar_cos);}
		if(order == 4) {dipolar_cos_pos = (4.375 * pow(dipolar_cos, 4) - 3.75 * pow(dipolar_cos, 2) + 0.375);}
		if(order == 5) {dipolar_cos_pos = (7.875 * pow(dipolar_cos, 5) - 8.75 * pow(dipolar_cos, 3) + 1.875 * dipolar_cos);}
		if(order == 6) {dipolar_cos_pos = (14.4375 * pow(dipolar_cos, 6) - 19.6875 * pow(dipolar_cos, 4) + 6.5625 * pow(dipolar_cos, 2) - 0.3145);}

		cos_bin = int((dipolar_cos_pos+1)/2 * segments_per_pi);
		entry = segments_per_pi * dist_bin + cos_bin;
		//if ((dist_bin < max_distance * segments_per_angstroem) && (cos_bin < segments_per_pi)) { 
		if(entry >= 0 && entry < int(max_distance * segments_per_angstroem * segments_per_pi)) {
			histogram[entry] += 1; 
		}
	}
}

void calc_donor_grid(double * coor, int * bond_table, int donor_count, double cutoff) {
	int i, j;
	double dist_sq, cutoff_sq = pow(cutoff, 2);

	for(i = 0; i < donor_count; i++) {
		bond_table[donor_count*i+i] = 0;

		for(j = i+1; j < donor_count; j++) {
			dist_sq = pow(coor[3*i]-coor[3*j],2) + pow(coor[3*i+1]-coor[3*j+1],2) + pow(coor[3*i+2]-coor[3*j+2],2);
			if(dist_sq <= cutoff) {
				bond_table[donor_count*i+j] = 1;
				bond_table[donor_count*j+i] = 1;
			}

			else {
				bond_table[donor_count*i+j] = 0;
				bond_table[donor_count*j+i] = 0;
			}
		}
	}
}

void calc_sum_Venn_MD_Cage_Single(double * mdcage_timeseries, double * dipoles, int n_particles, char * dataset1, char * dataset2, int maxshell1, int maxshell2) {
	int i;
	int shell1, shell2;

	for(i = 0; i < n_particles; i++) {
		shell1 = (int) dataset1[i] - 1;
		shell2 = (int) dataset2[i] - 1;

		(shell1 < maxshell1) ? (shell1 = shell1) : (shell1 = maxshell1);
		(shell2 < maxshell2) ? (shell2 = shell2) : (shell2 = maxshell2);

		mdcage_timeseries[shell1*(maxshell2+1)*3 + shell2*3 + 0] += dipoles[i*3 + 0];
		mdcage_timeseries[shell1*(maxshell2+1)*3 + shell2*3 + 1] += dipoles[i*3 + 1];
		mdcage_timeseries[shell1*(maxshell2+1)*3 + shell2*3 + 2] += dipoles[i*3 + 2];
	}
}

void separateCollectiveDipolesSpherically(double * coor, double * dipoles, int n_particles, double * aufpunkt, double * dip_inside, double * dip_outside, double cutoff) {
	int i;
	double dist_sq;

	for(i = 0; i < n_particles; i++) {
		dist_sq = pow((coor[3*i+0] - aufpunkt[0]), 2) + pow((coor[3*i+1] - aufpunkt[1]), 2) + pow((coor[3*i+2] - aufpunkt[2]), 2);

		if(dist_sq < cutoff*cutoff) {
			dip_inside[0] += dipoles[3*i+0];
			dip_inside[1] += dipoles[3*i+1];
			dip_inside[2] += dipoles[3*i+2];
		}

		else {
			dip_outside[0] += dipoles[3*i+0];
			dip_outside[1] += dipoles[3*i+1];
			dip_outside[2] += dipoles[3*i+2];
		}
	}
}

void correlateSingleVectorTS(double * timeseries, double * result, int correlation_length, int order = 1) {
	int i, j; 
	double v0x, v0y, v0z;
	double vtx, vty, vtz;
	double norm0, normt, value;

	for(i = 0; i < correlation_length; i++) {
		for(j = i; j < correlation_length; j++) {
			v0x = timeseries[3*i];
			v0y = timeseries[3*i+1];
			v0z = timeseries[3*i+2];

			vtx = timeseries[3*j];
			vty = timeseries[3*j+1];
			vtz = timeseries[3*j+2];

			norm0 = sqrt(v0x*v0x + v0y*v0y + v0z*v0z);
			normt = sqrt(vtx*vtx + vty*vty + vtz*vtz);

			if(norm0 != 0 && normt != 0) {
				value = (v0x*vtx + v0y*vty + v0z*vtz)/(norm0*normt);

				if(order == 1) {result[j-i] += (value);}
				if(order == 2) {result[j-i] += (1.5 * pow(value,2) - 0.5);}
				if(order == 3) {result[j-i] += (2.5 * pow(value, 3) - 1.5 * value);}
				if(order == 4) {result[j-i] += (4.375 * pow(value, 4) - 3.75 * pow(value, 2) + 0.375);}
				if(order == 5) {result[j-i] += (7.875 * pow(value, 5) - 8.75 * pow(value, 3) + 1.875 * value);}
				if(order == 6) {result[j-i] += (14.4375 * pow(value, 6) - 19.6875 * pow(value, 4) + 6.5625 * pow(value, 2) - 0.3145);}
			}
		}
	}

	for(i = 0; i < correlation_length; i++) {
		result[i] /= (correlation_length - i);
	}
}

void crossCorrelateSingleVectorTS(double * timeseries1, double * timeseries2, double * result, int correlation_length1, int correlation_length2, bool both_directions = 1, int order = 1) {

	int i, j; 
	double v0x, v0y, v0z;
	double vtx, vty, vtz;
	double norm0, normt, value;
	int correlation_length;

	(correlation_length1 < correlation_length2) ? (correlation_length = correlation_length1) : (correlation_length = correlation_length2);

	for(i = 0; i < correlation_length; i++) {
		for(j = i; j < correlation_length; j++) {
			v0x = timeseries1[3*i];
			v0y = timeseries1[3*i+1];
			v0z = timeseries1[3*i+2];

			vtx = timeseries2[3*j];
			vty = timeseries2[3*j+1];
			vtz = timeseries2[3*j+2];

			norm0 = sqrt(v0x*v0x + v0y*v0y + v0z*v0z);
			normt = sqrt(vtx*vtx + vty*vty + vtz*vtz);

			if(norm0 != 0 && normt != 0) {				
				value = (v0x*vtx + v0y*vty + v0z*vtz)/(norm0*normt);

				if(order == 1) {result[j-i] += (value);}
				if(order == 2) {result[j-i] += (1.5 * pow(value,2) - 0.5);}
				if(order == 3) {result[j-i] += (2.5 * pow(value, 3) - 1.5 * value);}
				if(order == 4) {result[j-i] += (4.375 * pow(value, 4) - 3.75 * pow(value, 2) + 0.375);}
				if(order == 5) {result[j-i] += (7.875 * pow(value, 5) - 8.75 * pow(value, 3) + 1.875 * value);}
				if(order == 6) {result[j-i] += (14.4375 * pow(value, 6) - 19.6875 * pow(value, 4) + 6.5625 * pow(value, 2) - 0.3145);}
			}
		}
	}

	if(both_directions) {
		for(i = 0; i < correlation_length; i++) {
			for(j = i; j < correlation_length; j++) {
				v0x = timeseries2[3*i];
				v0y = timeseries2[3*i+1];
				v0z = timeseries2[3*i+2];

				vtx = timeseries1[3*j];
				vty = timeseries1[3*j+1];
				vtz = timeseries1[3*j+2];

				norm0 = sqrt(v0x*v0x + v0y*v0y + v0z*v0z);
				normt = sqrt(vtx*vtx + vty*vty + vtz*vtz);

				if(norm0 != 0 && normt != 0) {
					value = (v0x*vtx + v0y*vty + v0z*vtz)/(norm0*normt);

					if(order == 1) {result[j-i] += (value);}
					if(order == 2) {result[j-i] += (1.5 * pow(value,2) - 0.5);}
					if(order == 3) {result[j-i] += (2.5 * pow(value, 3) - 1.5 * value);}
					if(order == 4) {result[j-i] += (4.375 * pow(value, 4) - 3.75 * pow(value, 2) + 0.375);}
					if(order == 5) {result[j-i] += (7.875 * pow(value, 5) - 8.75 * pow(value, 3) + 1.875 * value);}
					if(order == 6) {result[j-i] += (14.4375 * pow(value, 6) - 19.6875 * pow(value, 4) + 6.5625 * pow(value, 2) - 0.3145);}
				}
			}
		}
	}

	if(both_directions) {
		for(i = 0; i < correlation_length; i++) {
			result[i] /= (2*(correlation_length - i));
		}
	}

	else {
		for(i = 0; i < correlation_length; i++) {
			result[i] /= (correlation_length - i);
		}
	}
}

void correlateMultiVectorTS(double * timeseries, double * result, int number_of_particles, int correlation_length, int order = 1) {
	int i, j, k; 
	double v0x, v0y, v0z;
	double vtx, vty, vtz;
	double norm0, normt, value;

	for(k = 0; k < number_of_particles; k++) {
		for(i = 0; i < correlation_length; i++) {
			for(j = i; j < correlation_length; j++) {
				v0x = timeseries[3*correlation_length*k + 3*i + 0];
				v0y = timeseries[3*correlation_length*k + 3*i + 1];
				v0z = timeseries[3*correlation_length*k + 3*i + 2]; 

				vtx = timeseries[3*correlation_length*k + 3*j + 0];
				vty = timeseries[3*correlation_length*k + 3*j + 1];
				vtz = timeseries[3*correlation_length*k + 3*j + 2]; 

				norm0 = sqrt(v0x*v0x + v0y*v0y + v0z*v0z);
				normt = sqrt(vtx*vtx + vty*vty + vtz*vtz);

				if(norm0 != 0 && normt != 0) {
					value = (v0x*vtx + v0y*vty + v0z*vtz)/(norm0*normt);

					if(order == 1) {result[j-i] += (value);}
					if(order == 2) {result[j-i] += (1.5 * pow(value,2) - 0.5);}
					if(order == 3) {result[j-i] += (2.5 * pow(value, 3) - 1.5 * value);}
					if(order == 4) {result[j-i] += (4.375 * pow(value, 4) - 3.75 * pow(value, 2) + 0.375);}
					if(order == 5) {result[j-i] += (7.875 * pow(value, 5) - 8.75 * pow(value, 3) + 1.875 * value);}
					if(order == 6) {result[j-i] += (14.4375 * pow(value, 6) - 19.6875 * pow(value, 4) + 6.5625 * pow(value, 2) - 0.3145);}
				}
			}
		}
	}


	for(i = 0; i < correlation_length; i++) {
		result[i] /= ((correlation_length - i) * number_of_particles);
	}

}

void correlateMultiVectorShellwiseTS(double * timeseries, double * dataset, double * result, int number_of_particles, int correlation_length, int maxshell, int order = 1) {
	int i, j, k; 
	double v0x, v0y, v0z;
	double vtx, vty, vtz;
	double norm0, normt, value;
	int shell;
	vector<int> counter(maxshell);

	for(int i = 0; i < maxshell; i++) {counter[i] = 0;}

	for(k = 0; k < number_of_particles; k++) {
		for(i = 0; i < correlation_length; i++) {
			shell = dataset[i*number_of_particles + k] - 1; //TODO: Wirklich? dataset[timestep][particles]

			if(shell < maxshell) {
				counter.at(shell) += 1;

				for(j = i; j < correlation_length; j++) {
					v0x = timeseries[3*correlation_length*k + 3*i + 0];
					v0y = timeseries[3*correlation_length*k + 3*i + 1];
					v0z = timeseries[3*correlation_length*k + 3*i + 2]; 

					vtx = timeseries[3*correlation_length*k + 3*j + 0];
					vty = timeseries[3*correlation_length*k + 3*j + 1];
					vtz = timeseries[3*correlation_length*k + 3*j + 2]; 

					norm0 = sqrt(v0x*v0x + v0y*v0y + v0z*v0z);
					normt = sqrt(vtx*vtx + vty*vty + vtz*vtz);

					if(norm0 != 0 && normt != 0) {
						value = (v0x*vtx + v0y*vty + v0z*vtz)/(norm0*normt);

						if(order == 1) {result[shell*correlation_length + (j-i)] += (value);} //result[shell][timestep]
						if(order == 2) {result[shell*correlation_length + (j-i)] += (1.5 * pow(value,2) - 0.5);}
						if(order == 3) {result[shell*correlation_length + (j-i)] += (2.5 * pow(value, 3) - 1.5 * value);}
						if(order == 4) {result[shell*correlation_length + (j-i)] += (4.375 * pow(value, 4) - 3.75 * pow(value, 2) + 0.375);}
						if(order == 5) {result[shell*correlation_length + (j-i)] += (7.875 * pow(value, 5) - 8.75 * pow(value, 3) + 1.875 * value);}
						if(order == 6) {result[shell*correlation_length + (j-i)] += (14.4375 * pow(value, 6) - 19.6875 * pow(value, 4) + 6.5625 * pow(value, 2) - 0.3145);}
					}
				}
			}
		}
	}


	for(k = 0; k < maxshell; k++) {
		for(i = 0; i < correlation_length; i++) {
			if(counter.at(k) > 0) {
				result[k*correlation_length + i] /= ((counter.at(k))*(correlation_length - i)); //TODO: Wirklich? result[shell][timestep]
			}
		}
	}
}

void correlateMultiVectorVennShellwiseTS(double * timeseries, double * dataset1,  double * dataset2, double * result, int number_of_particles, int correlation_length, int maxshell1, int maxshell2, int order = 1) {
	int i, j, k, l; 
	double v0x, v0y, v0z;
	double vtx, vty, vtz;
	double norm0, normt, value;
	int shell1, shell2;
	vector<int> counter1(maxshell1);
	vector<int> counter2(maxshell2);

	for(int i = 0; i < maxshell1; i++) {counter1[i] = 0;}
	for(int i = 0; i < maxshell2; i++) {counter2[i] = 0;}

	for(k = 0; k < number_of_particles; k++) {
		for(i = 0; i < correlation_length; i++) {
			shell1 = dataset1[i*number_of_particles + k] - 1; //TODO: Wirklich? dataset[timestep][particles]
			shell2 = dataset2[i*number_of_particles + k] - 1;

			if((shell1 < maxshell1) && (shell2 < maxshell2)) {
				counter1.at(shell1) += 1;
				counter2.at(shell2) += 1;

				for(j = i; j < correlation_length; j++) {
					v0x = timeseries[3*correlation_length*k + 3*i + 0];
					v0y = timeseries[3*correlation_length*k + 3*i + 1];
					v0z = timeseries[3*correlation_length*k + 3*i + 2]; 

					vtx = timeseries[3*correlation_length*k + 3*j + 0];
					vty = timeseries[3*correlation_length*k + 3*j + 1];
					vtz = timeseries[3*correlation_length*k + 3*j + 2]; 

					norm0 = sqrt(v0x*v0x + v0y*v0y + v0z*v0z);
					normt = sqrt(vtx*vtx + vty*vty + vtz*vtz);

					if(norm0 != 0 && normt != 0) {
						value = (v0x*vtx + v0y*vty + v0z*vtz)/(norm0*normt);

						if(order == 1) {result[shell1*maxshell2*correlation_length + shell2*correlation_length + (j-i)] += (value);} //result[shell1][shell2][timestep]
						if(order == 2) {result[shell1*maxshell2*correlation_length + shell2*correlation_length + (j-i)] += (1.5 * pow(value,2) - 0.5);}
						if(order == 3) {result[shell1*maxshell2*correlation_length + shell2*correlation_length + (j-i)] += (2.5 * pow(value, 3) - 1.5 * value);}
						if(order == 4) {result[shell1*maxshell2*correlation_length + shell2*correlation_length + (j-i)] += (4.375 * pow(value, 4) - 3.75 * pow(value, 2) + 0.375);}
						if(order == 5) {result[shell1*maxshell2*correlation_length + shell2*correlation_length + (j-i)] += (7.875 * pow(value, 5) - 8.75 * pow(value, 3) + 1.875 * value);}
						if(order == 6) {result[shell1*maxshell2*correlation_length + shell2*correlation_length + (j-i)] += (14.4375 * pow(value, 6) - 19.6875 * pow(value, 4) + 6.5625 * pow(value, 2) - 0.3145);}
					}
				}
			}
		}
	}

	for(k = 0; k < maxshell1; k++) {
		for(l = 0; l < maxshell2; l++) {
			for(i = 0; i < correlation_length; i++) {
				if((counter1.at(k) + counter2.at(l)) > 0) {
					result[k*maxshell2*correlation_length + l*correlation_length + i] /= ((counter1.at(k) + counter2.at(l))*(correlation_length - i)); //TODO: Wirklich? result[shell1][shell2][timestep]
				}
			}
		}
	}
}

void crossCorrelateMultiVectorTS(double * timeseries1, double * timeseries2, double * result, int number_of_particles, int correlation_length1, int correlation_length2, bool both_directions = 1, int order = 1) {
	/*
	   ATTN: This cross-correlation function assumes equal amounts of particles in both trajectories
	*/
	int i, j, k; 
	double v0x, v0y, v0z;
	double vtx, vty, vtz;
	double norm0, normt, value;
	int correlation_length;

	(correlation_length1 < correlation_length2) ? (correlation_length = correlation_length1) : (correlation_length = correlation_length2);

	for(k = 0; k < number_of_particles; k++) {
		for(i = 0; i < correlation_length; i++) {
			for(j = i; j < correlation_length; j++) {
				v0x = timeseries1[3*correlation_length*k + 3*i + 0];
				v0y = timeseries1[3*correlation_length*k + 3*i + 1];
				v0z = timeseries1[3*correlation_length*k + 3*i + 2]; 

				vtx = timeseries2[3*correlation_length*k + 3*j + 0];
				vty = timeseries2[3*correlation_length*k + 3*j + 1];
				vtz = timeseries2[3*correlation_length*k + 3*j + 2]; 

				norm0 = sqrt(v0x*v0x + v0y*v0y + v0z*v0z);
				normt = sqrt(vtx*vtx + vty*vty + vtz*vtz);

				if(norm0 != 0 && normt != 0) {
					value = (v0x*vtx + v0y*vty + v0z*vtz)/(norm0*normt);

					if(order == 1) {result[j-i] += (value);}
					if(order == 2) {result[j-i] += (1.5 * pow(value,2) - 0.5);}
					if(order == 3) {result[j-i] += (2.5 * pow(value, 3) - 1.5 * value);}
					if(order == 4) {result[j-i] += (4.375 * pow(value, 4) - 3.75 * pow(value, 2) + 0.375);}
					if(order == 5) {result[j-i] += (7.875 * pow(value, 5) - 8.75 * pow(value, 3) + 1.875 * value);}
					if(order == 6) {result[j-i] += (14.4375 * pow(value, 6) - 19.6875 * pow(value, 4) + 6.5625 * pow(value, 2) - 0.3145);}
				}
			}
		}
	}

	if(both_directions) {
		for(k = 0; k < number_of_particles; k++) {
			for(i = 0; i < correlation_length; i++) {
				for(j = i; j < correlation_length; j++) {
					v0x = timeseries2[3*correlation_length*k + 3*i + 0];
					v0y = timeseries2[3*correlation_length*k + 3*i + 1];
					v0z = timeseries2[3*correlation_length*k + 3*i + 2]; 

					vtx = timeseries1[3*correlation_length*k + 3*j + 0];
					vty = timeseries1[3*correlation_length*k + 3*j + 1];
					vtz = timeseries1[3*correlation_length*k + 3*j + 2]; 

					norm0 = sqrt(v0x*v0x + v0y*v0y + v0z*v0z);
					normt = sqrt(vtx*vtx + vty*vty + vtz*vtz);

					if(norm0 != 0 && normt != 0) {
						value = (v0x*vtx + v0y*vty + v0z*vtz)/(norm0*normt);

						if(order == 1) {result[j-i] += (value);}
						if(order == 2) {result[j-i] += (1.5 * pow(value,2) - 0.5);}
						if(order == 3) {result[j-i] += (2.5 * pow(value, 3) - 1.5 * value);}
						if(order == 4) {result[j-i] += (4.375 * pow(value, 4) - 3.75 * pow(value, 2) + 0.375);}
						if(order == 5) {result[j-i] += (7.875 * pow(value, 5) - 8.75 * pow(value, 3) + 1.875 * value);}
						if(order == 6) {result[j-i] += (14.4375 * pow(value, 6) - 19.6875 * pow(value, 4) + 6.5625 * pow(value, 2) - 0.3145);}
					}
				}
			}
		}
	}

	if(both_directions) {
		for(i = 0; i < correlation_length; i++) {
			result[i] /= (2*number_of_particles*(correlation_length - i));
		}
	}

	else {
		for(i = 0; i < correlation_length; i++) {
			result[i] /= ((correlation_length - i) * number_of_particles);
		}
	}
}

void calcVanHoveSingleVector(double * timeseries, double * histogram, int correlation_length, int cos_segs) {
	int i, j;
	double vec_x_0, vec_y_0, vec_z_0;
	double vec_x_t, vec_y_t, vec_z_t;
	double norm_0, norm_t;
	double vec_cos;
	int cos_bin;

	for(i = 0; i < correlation_length; i++) {

		vec_x_0 = timeseries[3*i+0];
		vec_y_0 = timeseries[3*i+1];
		vec_z_0 = timeseries[3*i+2];

		norm_0 = sqrt(pow(vec_x_0, 2) + pow(vec_y_0, 2) + pow(vec_z_0, 2));

		for(j = i; j < correlation_length; j++) {

			vec_x_t = timeseries[3*j+0];
			vec_y_t = timeseries[3*j+1];
			vec_z_t = timeseries[3*j+2];

			norm_t = sqrt(pow(vec_x_t, 2) + pow(vec_y_t, 2) + pow(vec_z_t, 2));

			vec_cos = (vec_x_0 * vec_x_t + vec_y_0 * vec_y_t + vec_z_0 * vec_z_t)/(norm_0 * norm_t);
			cos_bin = int( (vec_cos + 1)/2 * cos_segs );

			if(cos_bin >= 0 && cos_bin < cos_segs) {
				histogram[(j-i)*cos_segs + cos_bin] += 1;
			}
		}
	}

	//TODO: Normalize the accumulation
}

void calcVanHoveMultiVector(double * timeseries, double * histogram, int n_particles, int correlation_length, int cos_segs) {
	int i, j, k;
	double vec_x_0, vec_y_0, vec_z_0;
	double vec_x_t, vec_y_t, vec_z_t;
	double norm_0, norm_t;
	double vec_cos;
	int cos_bin;

	for(k = 0; k < n_particles; k++) {
		for(i = 0; i < correlation_length; i++) {

			vec_x_0 = timeseries[correlation_length*3*k + 3*i + 0];
			vec_y_0 = timeseries[correlation_length*3*k + 3*i + 1];
			vec_z_0 = timeseries[correlation_length*3*k + 3*i + 2];

			norm_0 = sqrt(pow(vec_x_0, 2) + pow(vec_y_0, 2) + pow(vec_z_0, 2));

			for(j = i; j < correlation_length; j++) {

				vec_x_t = timeseries[correlation_length*3*k + 3*j + 0];
				vec_y_t = timeseries[correlation_length*3*k + 3*j + 1];
				vec_z_t = timeseries[correlation_length*3*k + 3*j + 2];

				norm_t = sqrt(pow(vec_x_t, 2) + pow(vec_y_t, 2) + pow(vec_z_t, 2));

				vec_cos = (vec_x_0 * vec_x_t + vec_y_0 * vec_y_t + vec_z_0 * vec_z_t)/(norm_0 * norm_t);
				cos_bin = int( (vec_cos + 1)/2 * cos_segs );

				if(cos_bin >= 0 && cos_bin < cos_segs) {
					histogram[(j-i)*cos_segs + cos_bin] += 1;
				}			
			}
		}
	}

	//TODO: Normalize the accumulation
}

void sort_collective_dip_NN_shells(char * ds, double * dip_wat, double * dip_shell, int n_particles) {
	int row, col, shell;

	for(row = 0; row < n_particles; row++) {
		for(col = 0; col < n_particles; col++) {
			shell = int(ds[row*n_particles + col]);
			dip_shell[shell*3 + 0] += dip_wat[row*3 + 0] / float(n_particles);
			dip_shell[shell*3 + 1] += dip_wat[row*3 + 1] / float(n_particles);
			dip_shell[shell*3 + 2] += dip_wat[row*3 + 2] / float(n_particles);
		}
	}
}

void sort_collective_dip_NN_shells_int(int * ds, double * dip_wat, double * dip_shell, int n_particles) {
	int row, col, shell;

	for(row = 0; row < n_particles; row++) {
		for(col = 0; col < n_particles; col++) {
			shell = int(ds[row*n_particles + col]);
			dip_shell[shell*3 + 0] += dip_wat[row*3 + 0] / float(n_particles);
			dip_shell[shell*3 + 1] += dip_wat[row*3 + 1] / float(n_particles);
			dip_shell[shell*3 + 2] += dip_wat[row*3 + 2] / float(n_particles);
		}
	}
}

void calc_dip_ten_collective(double * coor, int n_particles, double * results) {
	int i, j;
	double r, r_sq, r_cub;

	results[0] = 0;
	results[1] = 0;
	results[2] = 0;
	results[3] = 0;
	results[4] = 0;
	results[5] = 0;

	for(i = 0; i < n_particles; i++) {
		for(j = 0; j < n_particles; j++) {
			if(i != j) {
				r = sqrt(pow((coor[3*i+0]-coor[3*j+0]), 2) + pow((coor[3*i+1]-coor[3*j+1]), 2) + pow((coor[3*i+2]-coor[3*j+2]), 2));
				r_sq = pow(r, 2);
				r_cub = pow(r, 3);

				if(r_sq > 0) {
					results[0] += (1/r_cub) * ((3 * (coor[3*i+0] - coor[3*j+0]) * (coor[3*i+0] - coor[3*j+0]) / r_sq ) - 1); 	//xx
					results[1] += (1/r_cub) * ( 3 * (coor[3*i+0] - coor[3*j+0]) * (coor[3*i+1] - coor[3*j+1]) / r_sq ); 		//xy
					results[2] += (1/r_cub) * ( 3 * (coor[3*i+0] - coor[3*j+0]) * (coor[3*i+2] - coor[3*j+2]) / r_sq ); 		//xz
					results[3] += (1/r_cub) * ((3 * (coor[3*i+1] - coor[3*j+1]) * (coor[3*i+1] - coor[3*j+1]) / r_sq ) - 1); 	//yy
					results[4] += (1/r_cub) * ( 3 * (coor[3*i+1] - coor[3*j+1]) * (coor[3*i+2] - coor[3*j+2]) / r_sq ); 		//yz
					results[5] += (1/r_cub) * ((3 * (coor[3*i+2] - coor[3*j+2]) * (coor[3*i+2] - coor[3*j+2]) / r_sq ) - 1); 	//zz
				}
			}
		}
	}
}

void calc_dip_ten_collective_per_atom(int i, double * coor, int n_particles, double * results) {
	int j;
	double r, r_sq, r_cub;

	results[0] = 0;
	results[1] = 0;
	results[2] = 0;
	results[3] = 0;
	results[4] = 0;
	results[5] = 0;

	//double results_0 = 0, results_1 = 0, results_2 = 0, results_3 = 0, results_4 = 0, results_5 = 0;

	//#pragma omp parallel default(none) shared(n_particles, i, coor) private(j, r, r_sq, r_cub) reduction(+:results_0, results_1, results_2, results_3, results_4, results_5)
	{
		//#pragma omp for
		for(j = 0; j < n_particles; j++) {
			if(i != j) {
				r = sqrt(pow((coor[3*i+0]-coor[3*j+0]), 2) + pow((coor[3*i+1]-coor[3*j+1]), 2) + pow((coor[3*i+2]-coor[3*j+2]), 2));
				r_sq = pow(r, 2);
				r_cub = pow(r, 3);

				if(r_sq > 0) {
					results[0] += (1/r_cub) * ((3 * (coor[3*i+0] - coor[3*j+0]) * (coor[3*i+0] - coor[3*j+0]) / r_sq ) - 1); 	//xx
					results[1] += (1/r_cub) * ( 3 * (coor[3*i+0] - coor[3*j+0]) * (coor[3*i+1] - coor[3*j+1]) / r_sq ); 		//xy
					results[2] += (1/r_cub) * ( 3 * (coor[3*i+0] - coor[3*j+0]) * (coor[3*i+2] - coor[3*j+2]) / r_sq ); 		//xz
					results[3] += (1/r_cub) * ((3 * (coor[3*i+1] - coor[3*j+1]) * (coor[3*i+1] - coor[3*j+1]) / r_sq ) - 1); 	//yy
					results[4] += (1/r_cub) * ( 3 * (coor[3*i+1] - coor[3*j+1]) * (coor[3*i+2] - coor[3*j+2]) / r_sq ); 		//yz
					results[5] += (1/r_cub) * ((3 * (coor[3*i+2] - coor[3*j+2]) * (coor[3*i+2] - coor[3*j+2]) / r_sq ) - 1); 	//zz
					//results_0 += (1/r_cub) * ((3 * (coor[3*i+0] - coor[3*j+0]) * (coor[3*i+0] - coor[3*j+0]) / r_sq ) - 1);
					//results_1 += (1/r_cub) * ( 3 * (coor[3*i+0] - coor[3*j+0]) * (coor[3*i+1] - coor[3*j+1]) / r_sq );
					//results_2 += (1/r_cub) * ( 3 * (coor[3*i+0] - coor[3*j+0]) * (coor[3*i+2] - coor[3*j+2]) / r_sq );
					//results_3 += (1/r_cub) * ((3 * (coor[3*i+1] - coor[3*j+1]) * (coor[3*i+1] - coor[3*j+1]) / r_sq ) - 1);
					//results_4 += (1/r_cub) * ( 3 * (coor[3*i+1] - coor[3*j+1]) * (coor[3*i+2] - coor[3*j+2]) / r_sq );
					//results_5 += (1/r_cub) * ((3 * (coor[3*i+2] - coor[3*j+2]) * (coor[3*i+2] - coor[3*j+2]) / r_sq ) - 1);
				}
			}
		}
	}

	//results[0] = results_0;
	//results[1] = results_1;
	//results[2] = results_2;
	//results[3] = results_3;
	//results[4] = results_4;
	//results[5] = results_5;
}

void calc_dip_ten_collective_cross(double * coor1, int n_particles1, double * coor2, int n_particles2, double * results) {
	int i, j;
	double r, r_sq, r_cub;

	results[0] = 0;
	results[1] = 0;
	results[2] = 0;
	results[3] = 0;
	results[4] = 0;
	results[5] = 0;

	for(i = 0; i < n_particles1; i++) {
		for(j = 0; j < n_particles2; j++) {
			r = sqrt(pow((coor1[3*i+0]-coor2[3*j+0]), 2) + pow((coor1[3*i+1]-coor2[3*j+1]), 2) + pow((coor1[3*i+2]-coor2[3*j+2]), 2));
			r_sq = pow(r, 2);
			r_cub = pow(r, 3);

			if(r_sq > 0) {
				results[0] += (1/r_cub) * ((3 * (coor1[3*i+0] - coor2[3*j+0]) * (coor1[3*i+0] - coor2[3*j+0]) / r_sq ) - 1); 	//xx
				results[1] += (1/r_cub) * ( 3 * (coor1[3*i+0] - coor2[3*j+0]) * (coor1[3*i+1] - coor2[3*j+1]) / r_sq ); 	//xy
				results[2] += (1/r_cub) * ( 3 * (coor1[3*i+0] - coor2[3*j+0]) * (coor1[3*i+2] - coor2[3*j+2]) / r_sq ); 	//xz
				results[3] += (1/r_cub) * ((3 * (coor1[3*i+1] - coor2[3*j+1]) * (coor1[3*i+1] - coor2[3*j+1]) / r_sq ) - 1); 	//yy
				results[4] += (1/r_cub) * ( 3 * (coor1[3*i+1] - coor2[3*j+1]) * (coor1[3*i+2] - coor2[3*j+2]) / r_sq ); 	//yz
				results[5] += (1/r_cub) * ((3 * (coor1[3*i+2] - coor2[3*j+2]) * (coor1[3*i+2] - coor2[3*j+2]) / r_sq ) - 1); 	//zz
			}
		}
	}

	for(j = 0; j < n_particles2; j++) {
		for(i = 0; i < n_particles1; i++) {
			r = sqrt(pow((coor2[3*j+0]-coor1[3*i+0]), 2) + pow((coor2[3*j+1]-coor1[3*i+1]), 2) + pow((coor2[3*j+2]-coor1[3*i+2]), 2));
			r_sq = pow(r, 2);
			r_cub = pow(r, 3);

			if(r_sq > 0) {
				results[0] += (1/r_cub) * ((3 * (coor2[3*j+0] - coor1[3*i+0]) * (coor2[3*j+0] - coor1[3*i+0]) / r_sq ) - 1); 	//xx
				results[1] += (1/r_cub) * ( 3 * (coor2[3*j+0] - coor1[3*i+0]) * (coor2[3*j+1] - coor1[3*i+1]) / r_sq ); 	//xy
				results[2] += (1/r_cub) * ( 3 * (coor2[3*j+0] - coor1[3*i+0]) * (coor2[3*j+2] - coor1[3*i+2]) / r_sq ); 	//xz
				results[3] += (1/r_cub) * ((3 * (coor2[3*j+1] - coor1[3*i+1]) * (coor2[3*j+1] - coor1[3*i+1]) / r_sq ) - 1); 	//yy
				results[4] += (1/r_cub) * ( 3 * (coor2[3*j+1] - coor1[3*i+1]) * (coor2[3*j+2] - coor1[3*i+2]) / r_sq ); 	//yz
				results[5] += (1/r_cub) * ((3 * (coor2[3*j+2] - coor1[3*i+2]) * (coor2[3*j+2] - coor1[3*i+2]) / r_sq ) - 1); 	//zz
			}
		}
	}
}

void calc_dip_ten_collective_NNshellwise(double * coor, int n_particles, char * ds, int maxshell, double * results) {
	int i, j, shell;
	double r, r_sq, r_cub;

	for(i = 0; i < maxshell; i++) {
		results[i*6 + 0] = 0;
		results[i*6 + 1] = 0;
		results[i*6 + 2] = 0;
		results[i*6 + 3] = 0;
		results[i*6 + 4] = 0;
		results[i*6 + 5] = 0;
	}

	for(i = 0; i < n_particles; i++) {
		for(j = 0; j < n_particles; j++) {
			if(i != j) {
				(int(ds[i*n_particles + j]) < maxshell) ? (shell = int(ds[i*n_particles + j]) - 1) : (shell = maxshell - 1);

				r = sqrt(pow((coor[3*i+0]-coor[3*j+0]), 2) + pow((coor[3*i+1]-coor[3*j+1]), 2) + pow((coor[3*i+2]-coor[3*j+2]), 2));
				r_sq = pow(r, 2);
				r_cub = pow(r, 3);

				if(r_sq > 0) {
					results[shell*6 + 0] += (1/r_cub) * ((3 * (coor[3*i+0] - coor[3*j+0]) * (coor[3*i+0] - coor[3*j+0]) / r_sq ) - 1); 	//xx
					results[shell*6 + 1] += (1/r_cub) * ( 3 * (coor[3*i+0] - coor[3*j+0]) * (coor[3*i+1] - coor[3*j+1]) / r_sq ); 		//xy
					results[shell*6 + 2] += (1/r_cub) * ( 3 * (coor[3*i+0] - coor[3*j+0]) * (coor[3*i+2] - coor[3*j+2]) / r_sq ); 		//xz
					results[shell*6 + 3] += (1/r_cub) * ((3 * (coor[3*i+1] - coor[3*j+1]) * (coor[3*i+1] - coor[3*j+1]) / r_sq ) - 1); 	//yy
					results[shell*6 + 4] += (1/r_cub) * ( 3 * (coor[3*i+1] - coor[3*j+1]) * (coor[3*i+2] - coor[3*j+2]) / r_sq ); 		//yz
					results[shell*6 + 5] += (1/r_cub) * ((3 * (coor[3*i+2] - coor[3*j+2]) * (coor[3*i+2] - coor[3*j+2]) / r_sq ) - 1); 	//zz
				}
			}
		}
	}
}


void calc_dip_ten_collective_NNshellwise_self(double * coor, int * f2c, int n_particles, int n_particles_tot, char * ds, int ds_idx, int maxshell, double * results) {
	int i, j, pos_1, pos_2, shell;
	double r, r_sq, r_cub;

	for(i = 0; i < maxshell; i++) {
		results[i*6 + 0] = 0;
		results[i*6 + 1] = 0;
		results[i*6 + 2] = 0;
		results[i*6 + 3] = 0;
		results[i*6 + 4] = 0;
		results[i*6 + 5] = 0;
	}

	for(i = 0; i < n_particles; i++) {
		for(j = 0; j < n_particles; j++) {
			if(i != j) {
				pos_1 = f2c[i];
				pos_2 = f2c[j];
				//TODO: Debug below
				(int(ds[(pos_1+ds_idx)*n_particles_tot + (pos_2+ds_idx)]) < maxshell) ? (shell = int(ds[(pos_1+ds_idx)*n_particles_tot + (pos_2+ds_idx)]) - 1) : (shell = maxshell - 1);

				r = sqrt(pow((coor[3*i+0]-coor[3*j+0]), 2) + pow((coor[3*i+1]-coor[3*j+1]), 2) + pow((coor[3*i+2]-coor[3*j+2]), 2));
				r_sq = pow(r, 2);
				r_cub = pow(r, 3);

				if(r_sq > 0) {
					results[shell*6 + 0] += (1/r_cub) * ((3 * (coor[3*i+0] - coor[3*j+0]) * (coor[3*i+0] - coor[3*j+0]) / r_sq ) - 1); 	//xx
					results[shell*6 + 1] += (1/r_cub) * ( 3 * (coor[3*i+0] - coor[3*j+0]) * (coor[3*i+1] - coor[3*j+1]) / r_sq ); 		//xy
					results[shell*6 + 2] += (1/r_cub) * ( 3 * (coor[3*i+0] - coor[3*j+0]) * (coor[3*i+2] - coor[3*j+2]) / r_sq ); 		//xz
					results[shell*6 + 3] += (1/r_cub) * ((3 * (coor[3*i+1] - coor[3*j+1]) * (coor[3*i+1] - coor[3*j+1]) / r_sq ) - 1); 	//yy
					results[shell*6 + 4] += (1/r_cub) * ( 3 * (coor[3*i+1] - coor[3*j+1]) * (coor[3*i+2] - coor[3*j+2]) / r_sq ); 		//yz
					results[shell*6 + 5] += (1/r_cub) * ((3 * (coor[3*i+2] - coor[3*j+2]) * (coor[3*i+2] - coor[3*j+2]) / r_sq ) - 1); 	//zz
				}
			}
		}
	}
}


void calc_dip_ten_collective_NNshellwise_cross(double * coor1, double * coor2, int * f2c, int n_particles_1, int n_particles_2, int n_particles_tot, char * ds, int ds1_idx, int ds2_idx, int maxshell, double * results) {
	int i, j, pos_1, pos_2, shell;
	double r, r_sq, r_cub;

	for(i = 0; i < maxshell; i++) {
		results[i*6 + 0] = 0;
		results[i*6 + 1] = 0;
		results[i*6 + 2] = 0;
		results[i*6 + 3] = 0;
		results[i*6 + 4] = 0;
		results[i*6 + 5] = 0;
	}

	for(i = 0; i < n_particles_1; i++) {
		for(j = 0; j < n_particles_2; j++) {
			if(i != j) {
				pos_1 = f2c[i];
				pos_2 = f2c[j];
				//TODO: Debug below
				(int(ds[(pos_1+ds1_idx)*n_particles_tot + (pos_2+ds2_idx)]) < maxshell) ? (shell = int(ds[(pos_1+ds1_idx)*n_particles_tot + (pos_2+ds2_idx)]) - 1) : (shell = maxshell - 1);

				r = sqrt(pow((coor1[3*i+0]-coor2[3*j+0]), 2) + pow((coor1[3*i+1]-coor2[3*j+1]), 2) + pow((coor1[3*i+2]-coor2[3*j+2]), 2));
				r_sq = pow(r, 2);
				r_cub = pow(r, 3);

				if(r_sq > 0) {
					results[shell*6 + 0] += (1/r_cub) * ((3 * (coor1[3*i+0] - coor2[3*j+0]) * (coor1[3*i+0] - coor2[3*j+0]) / r_sq ) - 1); 	//xx
					results[shell*6 + 1] += (1/r_cub) * ( 3 * (coor1[3*i+0] - coor2[3*j+0]) * (coor1[3*i+1] - coor2[3*j+1]) / r_sq ); 	//xy
					results[shell*6 + 2] += (1/r_cub) * ( 3 * (coor1[3*i+0] - coor2[3*j+0]) * (coor1[3*i+2] - coor2[3*j+2]) / r_sq ); 	//xz
					results[shell*6 + 3] += (1/r_cub) * ((3 * (coor1[3*i+1] - coor2[3*j+1]) * (coor1[3*i+1] - coor2[3*j+1]) / r_sq ) - 1); 	//yy
					results[shell*6 + 4] += (1/r_cub) * ( 3 * (coor1[3*i+1] - coor2[3*j+1]) * (coor1[3*i+2] - coor2[3*j+2]) / r_sq ); 	//yz
					results[shell*6 + 5] += (1/r_cub) * ((3 * (coor1[3*i+2] - coor2[3*j+2]) * (coor1[3*i+2] - coor2[3*j+2]) / r_sq ) - 1); 	//zz

					//THE BELOW OPPOSITE-SITE CONTRIBUTIONS DO NOT NEED THEIR OWN DOUBLE LOOPS, DO THEY?
					results[shell*6 + 0] += (1/r_cub) * ((3 * (coor2[3*j+0] - coor1[3*i+0]) * (coor2[3*j+0] - coor1[3*i+0]) / r_sq ) - 1); 	//xx
					results[shell*6 + 1] += (1/r_cub) * ( 3 * (coor2[3*j+0] - coor1[3*i+0]) * (coor2[3*j+1] - coor1[3*i+1]) / r_sq ); 	//xy
					results[shell*6 + 2] += (1/r_cub) * ( 3 * (coor2[3*j+0] - coor1[3*i+0]) * (coor2[3*j+2] - coor1[3*i+2]) / r_sq ); 	//xz
					results[shell*6 + 3] += (1/r_cub) * ((3 * (coor2[3*j+1] - coor1[3*i+1]) * (coor2[3*j+1] - coor1[3*i+1]) / r_sq ) - 1); 	//yy
					results[shell*6 + 4] += (1/r_cub) * ( 3 * (coor2[3*j+1] - coor1[3*i+1]) * (coor2[3*j+2] - coor1[3*i+2]) / r_sq ); 	//yz
					results[shell*6 + 5] += (1/r_cub) * ((3 * (coor2[3*j+2] - coor1[3*i+2]) * (coor2[3*j+2] - coor1[3*i+2]) / r_sq ) - 1); 	//zz
				}
			}
		}
	}
}


void calc_dip_ten_collective_1Nshellwise_self(double * coor, int n_particles, char * ds, int maxshell, double * results) {
	int i, j, shell1, shell2, aux;
	double r, r_sq, r_cub;

	for(i = 0; i < maxshell; i++) {
		for(j = 0; j < maxshell; j++) {
			results[i*maxshell*6 + j*6 + 0] = 0;
			results[i*maxshell*6 + j*6 + 1] = 0;
			results[i*maxshell*6 + j*6 + 2] = 0;
			results[i*maxshell*6 + j*6 + 3] = 0;
			results[i*maxshell*6 + j*6 + 4] = 0;
			results[i*maxshell*6 + j*6 + 5] = 0;
		}
	}

	for(i = 0; i < n_particles; i++) {
		for(j = 0; j < n_particles; j++) {
			if(i != j) {
				(int(ds[i]) < maxshell) ? (shell1 = int(ds[i]) - 1) : (shell1 = maxshell - 1);
				(int(ds[j]) < maxshell) ? (shell2 = int(ds[j]) - 1) : (shell2 = maxshell - 1);

				if(shell1 > shell2) {
					aux = shell1;
					shell1 = shell2;
					shell2 = aux;
				}

				r = sqrt(pow((coor[3*i+0]-coor[3*j+0]), 2) + pow((coor[3*i+1]-coor[3*j+1]), 2) + pow((coor[3*i+2]-coor[3*j+2]), 2));
				r_sq = pow(r, 2);
				r_cub = pow(r, 3);

				if(r_sq > 0) {
					results[shell1*maxshell*6 + shell2*6 + 0] += (1/r_cub) * ((3 * (coor[3*i+0] - coor[3*j+0]) * (coor[3*i+0] - coor[3*j+0]) / r_sq ) - 1); 	//xx
					results[shell1*maxshell*6 + shell2*6 + 1] += (1/r_cub) * ( 3 * (coor[3*i+0] - coor[3*j+0]) * (coor[3*i+1] - coor[3*j+1]) / r_sq ); 		//xy
					results[shell1*maxshell*6 + shell2*6 + 2] += (1/r_cub) * ( 3 * (coor[3*i+0] - coor[3*j+0]) * (coor[3*i+2] - coor[3*j+2]) / r_sq ); 		//xz
					results[shell1*maxshell*6 + shell2*6 + 3] += (1/r_cub) * ((3 * (coor[3*i+1] - coor[3*j+1]) * (coor[3*i+1] - coor[3*j+1]) / r_sq ) - 1); 	//yy
					results[shell1*maxshell*6 + shell2*6 + 4] += (1/r_cub) * ( 3 * (coor[3*i+1] - coor[3*j+1]) * (coor[3*i+2] - coor[3*j+2]) / r_sq ); 		//yz
					results[shell1*maxshell*6 + shell2*6 + 5] += (1/r_cub) * ((3 * (coor[3*i+2] - coor[3*j+2]) * (coor[3*i+2] - coor[3*j+2]) / r_sq ) - 1); 	//zz
				}
			}
		}
	}
}


void calc_dip_ten_collective_1Nshellwise_cross(double * coor1, int n_particles1, double * coor2, int n_particles2, char * ds, int maxshell, double * results) {
	int i, j, shell;
	double r, r_sq, r_cub;

	for(i = 0; i < maxshell; i++) {
		results[i*6 + 0] = 0;
		results[i*6 + 1] = 0;
		results[i*6 + 2] = 0;
		results[i*6 + 3] = 0;
		results[i*6 + 4] = 0;
		results[i*6 + 5] = 0;
	}

	for(i = 0; i < n_particles1; i++) {
		for(j = 0; j < n_particles2; j++) {
			(int(ds[i]) < maxshell) ? (shell = int(ds[i]) - 1) : (shell = maxshell - 1);
			r = sqrt(pow((coor1[3*i+0]-coor2[3*j+0]), 2) + pow((coor1[3*i+1]-coor2[3*j+1]), 2) + pow((coor1[3*i+2]-coor2[3*j+2]), 2));
			r_sq = pow(r, 2);
			r_cub = pow(r, 3);

			if(r_sq > 0) {
				results[shell*6 + 0] += (1/r_cub) * ((3 * (coor1[3*i+0] - coor2[3*j+0]) * (coor1[3*i+0] - coor2[3*j+0]) / r_sq ) - 1); 	//xx
				results[shell*6 + 1] += (1/r_cub) * ( 3 * (coor1[3*i+0] - coor2[3*j+0]) * (coor1[3*i+1] - coor2[3*j+1]) / r_sq ); 	//xy
				results[shell*6 + 2] += (1/r_cub) * ( 3 * (coor1[3*i+0] - coor2[3*j+0]) * (coor1[3*i+2] - coor2[3*j+2]) / r_sq ); 	//xz
				results[shell*6 + 3] += (1/r_cub) * ((3 * (coor1[3*i+1] - coor2[3*j+1]) * (coor1[3*i+1] - coor2[3*j+1]) / r_sq ) - 1); 	//yy
				results[shell*6 + 4] += (1/r_cub) * ( 3 * (coor1[3*i+1] - coor2[3*j+1]) * (coor1[3*i+2] - coor2[3*j+2]) / r_sq ); 	//yz
				results[shell*6 + 5] += (1/r_cub) * ((3 * (coor1[3*i+2] - coor2[3*j+2]) * (coor1[3*i+2] - coor2[3*j+2]) / r_sq ) - 1); 	//zz

				results[shell*6 + 0] += (1/r_cub) * ((3 * (coor2[3*j+0] - coor1[3*i+0]) * (coor2[3*j+0] - coor1[3*i+0]) / r_sq ) - 1); 	//xx
				results[shell*6 + 1] += (1/r_cub) * ( 3 * (coor2[3*j+0] - coor1[3*i+0]) * (coor2[3*j+1] - coor1[3*i+1]) / r_sq ); 	//xy
				results[shell*6 + 2] += (1/r_cub) * ( 3 * (coor2[3*j+0] - coor1[3*i+0]) * (coor2[3*j+2] - coor1[3*i+2]) / r_sq ); 	//xz
				results[shell*6 + 3] += (1/r_cub) * ((3 * (coor2[3*j+1] - coor1[3*i+1]) * (coor2[3*j+1] - coor1[3*i+1]) / r_sq ) - 1); 	//yy
				results[shell*6 + 4] += (1/r_cub) * ( 3 * (coor2[3*j+1] - coor1[3*i+1]) * (coor2[3*j+2] - coor1[3*i+2]) / r_sq ); 	//yz
				results[shell*6 + 5] += (1/r_cub) * ((3 * (coor2[3*j+2] - coor1[3*i+2]) * (coor2[3*j+2] - coor1[3*i+2]) / r_sq ) - 1); 	//zz
			}
		}
	}
	/* DOESNT NEED ITS OWN DOUBLE LOOP; INCORPORATED ABOVE NOW
	for(j = 0; j < n_particles2; j++) {
		for(i = 0; i < n_particles1; i++) {
			(int(ds[i]) < maxshell) ? (shell = int(ds[i]) - 1) : (shell = maxshell - 1);
			r = sqrt(pow((coor2[3*j+0]-coor1[3*i+0]), 2) + pow((coor2[3*j+1]-coor1[3*i+1]), 2) + pow((coor2[3*j+2]-coor1[3*i+2]), 2));
			r_sq = pow(r, 2);
			r_cub = pow(r, 3);

			if(r_sq > 0) {
				results[shell*6 + 0] += (1/r_cub) * ((3 * (coor2[3*j+0] - coor1[3*i+0]) * (coor2[3*j+0] - coor1[3*i+0]) / r_sq ) - 1); 	//xx
				results[shell*6 + 1] += (1/r_cub) * ( 3 * (coor2[3*j+0] - coor1[3*i+0]) * (coor2[3*j+1] - coor1[3*i+1]) / r_sq ); 	//xy
				results[shell*6 + 2] += (1/r_cub) * ( 3 * (coor2[3*j+0] - coor1[3*i+0]) * (coor2[3*j+2] - coor1[3*i+2]) / r_sq ); 	//xz
				results[shell*6 + 3] += (1/r_cub) * ((3 * (coor2[3*j+1] - coor1[3*i+1]) * (coor2[3*j+1] - coor1[3*i+1]) / r_sq ) - 1); 	//yy
				results[shell*6 + 4] += (1/r_cub) * ( 3 * (coor2[3*j+1] - coor1[3*i+1]) * (coor2[3*j+2] - coor1[3*i+2]) / r_sq ); 	//yz
				results[shell*6 + 5] += (1/r_cub) * ((3 * (coor2[3*j+2] - coor1[3*i+2]) * (coor2[3*j+2] - coor1[3*i+2]) / r_sq ) - 1); 	//zz
			}
		}
	}
	*/
}


void calc_dip_ten_collective_shellwise(double * coor, int n_particles, char * ds, int maxshell, double * results) {
	int i, j, shell;
	double r, r_sq, r_cub;

	for(i = 0; i < maxshell; i++) {
		results[i*6 + 0] = 0;
		results[i*6 + 1] = 0;
		results[i*6 + 2] = 0;
		results[i*6 + 3] = 0;
		results[i*6 + 4] = 0;
		results[i*6 + 5] = 0;
	}

	for(i = 0; i < n_particles; i++) {
		for(j = 0; j < n_particles; j++) {
			if(i != j) {
				(int(ds[i]) < maxshell) ? (shell = int(ds[i])) : (shell = maxshell);

				r = sqrt(pow((coor[3*i+0]-coor[3*j+0]), 2) + pow((coor[3*i+1]-coor[3*j+1]), 2) + pow((coor[3*i+2]-coor[3*j+2]), 2));
				r_sq = pow(r, 2);
				r_cub = pow(r, 3);

				if(r_sq > 0) {
					results[shell*6 + 0] += (1/r_cub) * ((3 * (coor[3*i+0] - coor[3*j+0]) * (coor[3*i+0] - coor[3*j+0]) / r_sq ) - 1); 	//xx
					results[shell*6 + 1] += (1/r_cub) * ( 3 * (coor[3*i+0] - coor[3*j+0]) * (coor[3*i+1] - coor[3*j+1]) / r_sq ); 		//xy
					results[shell*6 + 2] += (1/r_cub) * ( 3 * (coor[3*i+0] - coor[3*j+0]) * (coor[3*i+2] - coor[3*j+2]) / r_sq ); 		//xz
					results[shell*6 + 3] += (1/r_cub) * ((3 * (coor[3*i+1] - coor[3*j+1]) * (coor[3*i+1] - coor[3*j+1]) / r_sq ) - 1); 	//yy
					results[shell*6 + 4] += (1/r_cub) * ( 3 * (coor[3*i+1] - coor[3*j+1]) * (coor[3*i+2] - coor[3*j+2]) / r_sq ); 		//yz
					results[shell*6 + 5] += (1/r_cub) * ((3 * (coor[3*i+2] - coor[3*j+2]) * (coor[3*i+2] - coor[3*j+2]) / r_sq ) - 1); 	//zz
				}
			}
		}
	}
} 

void calc_dip_ten_collective_vennshellwise(double * coor, int n_particles, char * ds1, char * ds2, int maxshell1, int maxshell2, double * results) {
	int i, j, shell1, shell2;
	double r, r_sq, r_cub;

	for(i = 0; i < maxshell1; i++) {
		for(j = 0; j < maxshell2; j++) {
			results[j*6*maxshell1 + i*6 + 0] = 0;
			results[j*6*maxshell1 + i*6 + 1] = 0;
			results[j*6*maxshell1 + i*6 + 2] = 0;
			results[j*6*maxshell1 + i*6 + 3] = 0;
			results[j*6*maxshell1 + i*6 + 4] = 0;
			results[j*6*maxshell1 + i*6 + 5] = 0;
		}
	}

	for(i = 0; i < n_particles; i++) {
		for(j = 0; j < n_particles; j++) {
			if(i != j) {
				(int(ds1[i]) < maxshell1) ? (shell1 = int(ds1[i])) : (shell1 = maxshell1);
				(int(ds2[i]) < maxshell2) ? (shell2 = int(ds2[i])) : (shell2 = maxshell2);

				r = sqrt(pow((coor[3*i+0]-coor[3*j+0]), 2) + pow((coor[3*i+1]-coor[3*j+1]), 2) + pow((coor[3*i+2]-coor[3*j+2]), 2));
				r_sq = pow(r, 2);
				r_cub = pow(r, 3);

				if(r_sq > 0) {
					results[shell2*maxshell1*6 + shell1*6 + 0] += (1/r_cub) * ((3 * (coor[3*i+0] - coor[3*j+0]) * (coor[3*i+0] - coor[3*j+0]) / r_sq ) - 1); 	//xx
					results[shell2*maxshell1*6 + shell1*6 + 1] += (1/r_cub) * ( 3 * (coor[3*i+0] - coor[3*j+0]) * (coor[3*i+1] - coor[3*j+1]) / r_sq ); 		//xy
					results[shell2*maxshell1*6 + shell1*6 + 2] += (1/r_cub) * ( 3 * (coor[3*i+0] - coor[3*j+0]) * (coor[3*i+2] - coor[3*j+2]) / r_sq ); 		//xz
					results[shell2*maxshell1*6 + shell1*6 + 3] += (1/r_cub) * ((3 * (coor[3*i+1] - coor[3*j+1]) * (coor[3*i+1] - coor[3*j+1]) / r_sq ) - 1); 	//yy
					results[shell2*maxshell1*6 + shell1*6 + 4] += (1/r_cub) * ( 3 * (coor[3*i+1] - coor[3*j+1]) * (coor[3*i+2] - coor[3*j+2]) / r_sq ); 		//yz
					results[shell2*maxshell1*6 + shell1*6 + 5] += (1/r_cub) * ((3 * (coor[3*i+2] - coor[3*j+2]) * (coor[3*i+2] - coor[3*j+2]) / r_sq ) - 1); 	//zz
				}
			}
		}
	}
} 

void calc_distance_delauny_mindist(int * delauny_matrix, double * coor, int n_particles, double boxlength, int number_of_shells, double bin_width) {
	int particle_1, particle_2, shell;
	double boxl_half   = boxlength / 2.0;
	double dx, dy, dz;
	double dist;

	for(particle_1 = 0; particle_1 < n_particles; particle_1++) {
		for(particle_2 = 0; particle_2 < n_particles; particle_2++) {
			if (particle_1 == particle_2) {
				delauny_matrix[particle_1*n_particles + particle_2] = 0;
			}

			else {
				dx = coor[particle_2*3+0] - coor[particle_1*3+0];
				if(dx >   boxl_half) {dx -= boxlength;}
				if(dx <= -boxl_half) {dx += boxlength;}

				dy = coor[particle_2*3+1] - coor[particle_1*3+1];
				if(dy >   boxl_half) {dy -= boxlength;}
				if(dy <= -boxl_half) {dy += boxlength;}

				dz = coor[particle_2*3+2] - coor[particle_1*3+2];
				if(dz >   boxl_half) {dz -= boxlength;}
				if(dz <= -boxl_half) {dz += boxlength;}

				dist  = sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2));
				
				shell = int( (dist + (bin_width/2.0))/bin_width );
				if(shell >= number_of_shells) {delauny_matrix[particle_1*n_particles + particle_2] = -1;}
				else                          {delauny_matrix[particle_1*n_particles + particle_2] = shell;}
			}
		}
	}
}

void calc_distance_delauny(int * delauny_matrix, double * coor, int n_particles, int number_of_shells, double bin_width) {
	int particle_1, particle_2, shell;
	double dx, dy, dz;
	double dist;

	for(particle_1 = 0; particle_1 < n_particles; particle_1++) {
		for(particle_2 = 0; particle_2 < n_particles; particle_2++) {
			if (particle_1 == particle_2) {
				delauny_matrix[particle_1*n_particles + particle_2] = 0;
			}

			else {
				dx = coor[particle_2*3+0] - coor[particle_1*3+0];
				dy = coor[particle_2*3+1] - coor[particle_1*3+1];
				dz = coor[particle_2*3+2] - coor[particle_1*3+2];
				dist  = sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2));
				
				shell = int( (dist + (bin_width/2.0))/bin_width );
				if(shell >= number_of_shells) {delauny_matrix[particle_1*n_particles + particle_2] = -1;}
				else                          {delauny_matrix[particle_1*n_particles + particle_2] = shell;}
			}
		}
	}
}	

void calc_min_dist_tesselation(double * dataset, double * coor_core, double * coor_surround, int n_core, int n_surround, double binwidth) {
	int i, j, bin;
	double dist_sq, dist_sq_min;

	for(i = 0; i < n_surround; i++) {

		dist_sq_min = pow((coor_surround[3*i] - coor_core[0]), 2) + pow((coor_surround[3*i+1] - coor_core[1]), 2) + pow((coor_surround[3*i+2] - coor_core[2]), 2);

		for(j = 1; j < n_core; j++) {
			dist_sq = pow((coor_surround[3*i] - coor_core[3*j]), 2) + pow((coor_surround[3*i+1] - coor_core[3*j+1]), 2) + pow((coor_surround[3*i+2] - coor_core[3*j+2]), 2);
			if(dist_sq < dist_sq_min) {
				dist_sq = dist_sq_min;
			}
		}

		bin = int(sqrt(dist_sq_min)/binwidth) + 1;
		dataset[i] = bin; 
	}
}

void construct_relay_matrix(double * coor, double * inv_atom_polarizabilities, double * matrix, int n_atoms) {
	int i, j;
	double dist, dist_3, dist_5;
	double dist_x, dist_y, dist_z;

	// Could this be calculated as a quasi-triangular matrix?

	for(i = 0; i < n_atoms; i++) {
		for(j = 0; j < n_atoms; j++) {
			if(i == j) {
				matrix[((3*j+0)*3*n_atoms) + (3*i+0)] = inv_atom_polarizabilities[3*3*i + 3*0 + 0];
				matrix[((3*j+0)*3*n_atoms) + (3*i+1)] = inv_atom_polarizabilities[3*3*i + 3*0 + 1];
				matrix[((3*j+0)*3*n_atoms) + (3*i+2)] = inv_atom_polarizabilities[3*3*i + 3*0 + 2];
				matrix[((3*j+1)*3*n_atoms) + (3*i+0)] = inv_atom_polarizabilities[3*3*i + 3*1 + 0];
				matrix[((3*j+1)*3*n_atoms) + (3*i+1)] = inv_atom_polarizabilities[3*3*i + 3*1 + 1];
				matrix[((3*j+1)*3*n_atoms) + (3*i+2)] = inv_atom_polarizabilities[3*3*i + 3*1 + 2];
				matrix[((3*j+2)*3*n_atoms) + (3*i+0)] = inv_atom_polarizabilities[3*3*i + 3*2 + 0];
				matrix[((3*j+2)*3*n_atoms) + (3*i+1)] = inv_atom_polarizabilities[3*3*i + 3*2 + 1];
				matrix[((3*j+2)*3*n_atoms) + (3*i+2)] = inv_atom_polarizabilities[3*3*i + 3*2 + 2];
			}

			else {
				dist = sqrt( pow((coor[3*i+0] - coor[3*j+0]), 2) + pow((coor[3*i+1] - coor[3*j+1]), 2) + pow((coor[3*i+2] - coor[3*j+2]), 2) );
				dist_5 = pow(dist, 5);
				dist_3 = pow(dist, 3);

				dist_x = coor[3*i + 0] - coor[3*j + 0];
				dist_y = coor[3*i + 1] - coor[3*j + 1];
				dist_z = coor[3*i + 2] - coor[3*j + 2];

				matrix[((3*j+0)*3*n_atoms) + (3*i+0)] = 1.0/dist_3 - (3 * (dist_x * dist_x) / dist_5);
				matrix[((3*j+0)*3*n_atoms) + (3*i+1)] =               3 * (dist_x * dist_y) / dist_5;
				matrix[((3*j+0)*3*n_atoms) + (3*i+2)] =               3 * (dist_x * dist_z) / dist_5;
				matrix[((3*j+1)*3*n_atoms) + (3*i+0)] =               3 * (dist_y * dist_x) / dist_5;
				matrix[((3*j+1)*3*n_atoms) + (3*i+1)] = 1.0/dist_3 - (3 * (dist_y * dist_y) / dist_5);
				matrix[((3*j+1)*3*n_atoms) + (3*i+2)] =               3 * (dist_y * dist_z) / dist_5;
				matrix[((3*j+2)*3*n_atoms) + (3*i+0)] =               3 * (dist_z * dist_x) / dist_5;
				matrix[((3*j+2)*3*n_atoms) + (3*i+1)] =               3 * (dist_z * dist_y) / dist_5;
				matrix[((3*j+2)*3*n_atoms) + (3*i+2)] = 1.0/dist_3 - (3 * (dist_z * dist_z) / dist_5);
			}
		}
	}
}

/******************************
 *  Daniel NOE helpers        *
 ******************************/

void dipten_double_loop_(double * coor_1, double * coor_2, double * dipt_t, int n_particles_1, int n_particles_2, int only_different_nuclei) {
	double dist_sq, dist_2, dist_3, distvec[3];
	int i,j,index;

	index = 0;
	{
		for(i = 0; i < n_particles_1; i++) {
			for(j = i+only_different_nuclei; j < n_particles_2; j++) {
				distvec[0] = coor_2[3*j+0] - coor_1[3*i+0];
				distvec[1] = coor_2[3*j+1] - coor_1[3*i+1];
				distvec[2] = coor_2[3*j+2] - coor_1[3*i+2];
				dist_sq = distvec[0]*distvec[0] + distvec[1]*distvec[1] + distvec[2]*distvec[2];
				dist_2 = 1.0 / dist_sq;
				dist_3 = pow(dist_2, 1.5);

				dipt_t[6*index+0] = 3*dist_3*(distvec[0]*distvec[0]*dist_2-1);
				dipt_t[6*index+1] = 3*dist_3*(distvec[1]*distvec[1]*dist_2-1);
				dipt_t[6*index+2] = 3*dist_3*(distvec[2]*distvec[2]*dist_2-1);
				dipt_t[6*index+3] = 3*dist_3*(distvec[0]*distvec[1]*dist_2)*2;
				dipt_t[6*index+4] = 3*dist_3*(distvec[0]*distvec[2]*dist_2)*2;
				dipt_t[6*index+5] = 3*dist_3*(distvec[1]*distvec[2]*dist_2)*2;

				index += 1;
			}
		}
	}
}

void dipten_double_loop2_(double * coor_1, double * coor_2, double * dipt_t, int n_particles_1, int n_particles_2, int only_different_nuclei) {
	double dist_sq, dist_2, dist_3, distvec[3];
	int i,j,index;

	index = 0;
	for(i = 0; i < n_particles_1; i++) {
		for(j = i+only_different_nuclei; j < n_particles_2; j++) {
			distvec[0] = coor_2[3*i+0] - coor_1[3*j+0];
			distvec[1] = coor_2[3*i+1] - coor_1[3*j+1];
			distvec[2] = coor_2[3*i+2] - coor_1[3*j+2];
			dist_sq = distvec[0]*distvec[0] + distvec[1]*distvec[1] + distvec[2]*distvec[2];
			dist_2 = 1.0 / dist_sq;
			dist_3 = pow(dist_2, 1.5);

			dipt_t[6*index+0] = 3*dist_3*(distvec[0]*distvec[0]*dist_2-1);
			dipt_t[6*index+1] = 3*dist_3*(distvec[1]*distvec[1]*dist_2-1);
			dipt_t[6*index+2] = 3*dist_3*(distvec[2]*distvec[2]*dist_2-1);
			dipt_t[6*index+3] = 3*dist_3*(distvec[0]*distvec[1]*dist_2)*2;
			dipt_t[6*index+4] = 3*dist_3*(distvec[0]*distvec[2]*dist_2)*2;
			dipt_t[6*index+5] = 3*dist_3*(distvec[1]*distvec[2]*dist_2)*2;

			index += 1;
		}
	}
}


void pairiter_loop_(int pairlist_len, int * pairlist_p1, int * pairlist_p2, double * correlation_list, int n_pairs_h, int apr_emim_h, int * emim_h, int emim_h_len, double * run, double * dipt_0, double * dipt_t, int * bins, int max_distance) {
	int i, j, index, pair;
	double dist_sq, dist_2, dist_3, distvec[3];

	for(pair = 0; pair < pairlist_len; pair++) {
		double pair1list[apr_emim_h * 3];
		double pair2list[apr_emim_h * 3];

		index = 0;
		for(i = 0; i < emim_h_len; i++) {
			pair1list[index*3 + 0] = run[pairlist_p1[pair]*emim_h_len*3 + emim_h[i]*3 + 0];
			pair1list[index*3 + 1] = run[pairlist_p1[pair]*emim_h_len*3 + emim_h[i]*3 + 1];
			pair1list[index*3 + 2] = run[pairlist_p1[pair]*emim_h_len*3 + emim_h[i]*3 + 2];

			pair2list[index*3 + 0] = run[pairlist_p2[pair]*emim_h_len*3 + emim_h[i]*3 + 0];
			pair2list[index*3 + 1] = run[pairlist_p2[pair]*emim_h_len*3 + emim_h[i]*3 + 1];
			pair2list[index*3 + 2] = run[pairlist_p2[pair]*emim_h_len*3 + emim_h[i]*3 + 2];
			index++;
		}

		index = 0;
		for(i = 0; i < apr_emim_h - 1; i++) {
			for(j = i+1; j < apr_emim_h; j++) { //COMMIT 1: COMMENT THIS, UNCOMMENT BELOW
			//for(j = i; j < apr_emim_h; j++) {	
				distvec[0] = run[pairlist_p1[pair]*emim_h_len*3 + emim_h[j]*3 + 0] - run[pairlist_p1[pair]*emim_h_len*3 + emim_h[j]*3 + 0]; //TODO: Correct coordinate variable?
				distvec[1] = run[pairlist_p1[pair]*emim_h_len*3 + emim_h[j]*3 + 1] - run[pairlist_p1[pair]*emim_h_len*3 + emim_h[j]*3 + 1];
				distvec[2] = run[pairlist_p1[pair]*emim_h_len*3 + emim_h[j]*3 + 2] - run[pairlist_p1[pair]*emim_h_len*3 + emim_h[j]*3 + 2];

				dist_sq = distvec[0] * distvec[0] + distvec[1] * distvec[1] + distvec[2] * distvec[2];
				dist_2 = 1.0 / dist_sq;
				dist_3 = pow(dist_2, 1.5);

				dipt_t[6*i + 0] = dist_3 * (dist_2 * distvec[0] * distvec[0] - 1);
				dipt_t[6*i + 1] = dist_3 * (dist_2 * distvec[1] * distvec[1] - 1);
				dipt_t[6*i + 2] = dist_3 * (dist_2 * distvec[2] * distvec[2] - 1);
				dipt_t[6*i + 3] = dist_3 * (dist_2 * distvec[0] * distvec[1]) * 2;
				dipt_t[6*i + 4] = dist_3 * (dist_2 * distvec[0] * distvec[2]) * 2;
				dipt_t[6*i + 5] = dist_3 * (dist_2 * distvec[1] * distvec[2]) * 2;
			}
		}

		for(i = 0; i < n_pairs_h; i++) {
			correlation_list[i*max_distance + bins[pair*n_pairs_h + i]] += dipt_0[pair*n_pairs_h*6 + 6*i + 0] * dipt_t[6*i + 0];
			correlation_list[i*max_distance + bins[pair*n_pairs_h + i]] += dipt_0[pair*n_pairs_h*6 + 6*i + 1] * dipt_t[6*i + 1];
			correlation_list[i*max_distance + bins[pair*n_pairs_h + i]] += dipt_0[pair*n_pairs_h*6 + 6*i + 2] * dipt_t[6*i + 2];
			correlation_list[i*max_distance + bins[pair*n_pairs_h + i]] += dipt_0[pair*n_pairs_h*6 + 6*i + 3] * dipt_t[6*i + 3];
			correlation_list[i*max_distance + bins[pair*n_pairs_h + i]] += dipt_0[pair*n_pairs_h*6 + 6*i + 4] * dipt_t[6*i + 4];
			correlation_list[i*max_distance + bins[pair*n_pairs_h + i]] += dipt_0[pair*n_pairs_h*6 + 6*i + 5] * dipt_t[6*i + 5];
		}
	}
}

/******************************/
