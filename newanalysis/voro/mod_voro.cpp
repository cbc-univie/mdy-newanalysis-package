
#include <voro++/voro++.hh>
#include <vector>
#include <set>
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <iostream>
#include "mod_voro.h"

// void _calcTessellation(double *xyz_ptr, float boxlength, int32_t *f2c, int natoms, int nmolecules, int maxshell, char *ds_ptr, int32_t *corelist, int ncore)
// {
// //         std::cout << "_calcTessellation()" << std::endl;
//         int nblocks=25;

// 	double **xyz = new double*[natoms];
// 	for (int i=0;i<natoms;i++) 
// 	  xyz[i] = &(xyz_ptr[i*3]);

//         char **ds = new char*[nmolecules];
//         for (int i=0; i<nmolecules; i++)
//             ds[i]=&(ds_ptr[i*nmolecules]);
        
//         for (int i=0; i<nmolecules; i++) {
//             for (int j=0; j<nmolecules; j++) {
//                 if (i==j) ds[i][j]=0;
//                 else ds[i][j]=-1;
//             }
//         }

//         voro::container con(0,boxlength,0,boxlength,0,boxlength,nblocks,nblocks,nblocks,true,true,true,8);

//         std::vector<std::set<int> > neighbours;
        
//         neighbours.resize(nmolecules);
        
//         // add particle coordinates to the container
//         for (int i=0;i<natoms;i++) con.put(i,xyz[i][0],xyz[i][1],xyz[i][2]);
        
//         // calculate tessellation
//         voro::voronoicell_neighbor c;
//         voro::c_loop_all vl(con);
//         // neighbour list
//         std::vector<int> nb;
//         if (vl.start()) do {
//             con.compute_cell(c,vl); 
//             c.neighbors(nb);
//             for (unsigned int i=0;i<nb.size();i++) {
// // 		    con.id[vl.ijk][vl.q]                 ID of current particle
// // 		    nb[i] 				 current neighbor
// // 		    insert neighbours
//                 neighbours[f2c[con.id[vl.ijk][vl.q]]].insert(f2c[nb[i]]);
//             }
//             nb.clear();
//         }
//         while (vl.inc());

//         // clear container
//         con.clear();
        
//         std::vector<int> currShell;
//         for (int i=0; i<ncore; i++)
//         {
//                 currShell.clear();
//                 currShell.push_back(corelist[i]);

//                 for (int k=1; k <= maxshell; k++)
//                 {
//                         std::vector<int> prevShell(currShell);
//                         currShell.clear();
//                         for (unsigned int j=0; j < prevShell.size(); j++)
//                         {
//                                 for (std::set<int>::iterator it=neighbours[prevShell[j]].begin(); it != neighbours[prevShell[j]].end(); it++)
//                                 {
//                                         if (ds[corelist[i]][*it]==-1)
//                                         {
//                                                 ds[corelist[i]][*it]=k;
//                                                 currShell.push_back(*it);
//                                         }
//                                 }
//                         }
//                 }
//         }
//         neighbours.clear();
//         currShell.clear();
//         delete[] xyz;
//         delete[] ds;
// }

void _calcTessellation(double *xyz_ptr, float boxlength, int32_t *f2c, int natoms, int nmolecules, int maxshell, char *ds_ptr, int32_t *corelist, int ncore)
{
//         std::cout << "_calcTessellation()" << std::endl;
        int nblocks=25;
        

	double **xyz = new double*[natoms];
	for (int i=0;i<natoms;i++) 
	  xyz[i] = &(xyz_ptr[i*3]);

        char **ds = new char*[nmolecules];
        for (int i=0; i<nmolecules; i++)
            ds[i]=&(ds_ptr[i*nmolecules]);
        
        for (int i=0; i<nmolecules; i++) {
            for (int j=0; j<nmolecules; j++) {
                if (i==j) ds[i][j]=0;
                else ds[i][j]=-1;
            }
        }

        voro::container con(0,boxlength,0,boxlength,0,boxlength,nblocks,nblocks,nblocks,true,true,true,8);

        std::vector<std::set<int> > neighbours;
        
        neighbours.resize(nmolecules);
        
        // add particle coordinates to the container
        for (int i=0;i<natoms;i++) con.put(i,xyz[i][0],xyz[i][1],xyz[i][2]);
        
        // calculate tessellation
        voro::voronoicell_neighbor c;
        voro::c_loop_all vl(con);
        // neighbour list
        std::vector<int> nb;
        if (vl.start()) do {
            con.compute_cell(c,vl); 
            c.neighbors(nb);
            for (unsigned int i=0;i<nb.size();i++) {
// 		    con.id[vl.ijk][vl.q]                 ID of current particle
// 		    nb[i] 				 current neighbor
// 		    insert neighbours
                neighbours[f2c[con.id[vl.ijk][vl.q]]].insert(f2c[nb[i]]);
            }
            nb.clear();
        }
        while (vl.inc());

        // clear container
        con.clear();
        
        std::vector<int> currShell;
        for (int i=0; i<ncore; i++)
        {
                currShell.clear();
                currShell.push_back(corelist[i]);

                for (int k=1; k <= maxshell; k++)
                {
                        std::vector<int> prevShell(currShell);
                        currShell.clear();
                        for (unsigned int j=0; j < prevShell.size(); j++)
                        {
                                for (std::set<int>::iterator it=neighbours[prevShell[j]].begin(); it != neighbours[prevShell[j]].end(); it++)
                                {
                                        if (ds[corelist[i]][*it]==-1)
                                        {
                                                ds[corelist[i]][*it]=k;
                                                currShell.push_back(*it);
                                        }
                                }
                        }
                }
        }
        neighbours.clear();
        currShell.clear();
        delete[] xyz;
        delete[] ds;
}

void _calcTessellationVolSurf(double *xyz_ptr, float boxlength, int32_t *f2c, int natoms, int nmolecules, int maxshell, char *ds_ptr, int32_t *corelist, int ncore, float* vols, float* face_area_ptr)
{
//         std::cout << "_calcTessellation()" << std::endl;
        int nblocks=25;

	double **xyz = new double*[natoms];
	for (int i=0;i<natoms;i++) 
	  xyz[i] = &(xyz_ptr[i*3]);
	
        char **ds = new char*[nmolecules];
        for (int i=0; i<nmolecules; i++) {
            ds[i]=&(ds_ptr[i*nmolecules]);
	    for (int j=0; j<nmolecules; j++) {
                if (i==j) ds[i][j]=0;
                else ds[i][j]=-1;
            }
	    vols[i]=0.0;
	}
        
	float** face_areas = new float*[nmolecules];
	for (int i=0; i<nmolecules; i++) {
	  face_areas[i]=&(face_area_ptr[i*nmolecules]);
	  for (int j=0; j<nmolecules; j++)
	    face_areas[i][j]=0.0;
	}

        voro::container con(0,boxlength,0,boxlength,0,boxlength,nblocks,nblocks,nblocks,true,true,true,8);

        std::vector<std::set<int> > neighbours;
        
        neighbours.resize(nmolecules);
        
        // add particle coordinates to the container
        for (int i=0;i<natoms;i++) con.put(i,xyz[i][0],xyz[i][1],xyz[i][2]);
        
        // calculate tessellation
        voro::voronoicell_neighbor c;
        voro::c_loop_all vl(con);
        // neighbour list
        std::vector<int> nb;
	// face areas --- order is the same as in neighbor list
	std::vector<double> fa;

        if (vl.start()) do {
            con.compute_cell(c,vl); 
            c.neighbors(nb);
	    c.face_areas(fa);
	    vols[f2c[con.id[vl.ijk][vl.q]]] += c.volume();
            for (unsigned int i=0;i<nb.size();i++) {
// 		    con.id[vl.ijk][vl.q]                 ID of current particle
// 		    nb[i] 				 current neighbor
// 		    insert neighbours
                neighbours[f2c[con.id[vl.ijk][vl.q]]].insert(f2c[nb[i]]);
		if (f2c[con.id[vl.ijk][vl.q]] != f2c[nb[i]])
		  {
		    face_areas[f2c[con.id[vl.ijk][vl.q]]][f2c[nb[i]]]+=fa[i];
		  }
            }
            nb.clear();
	    fa.clear();
        }
        while (vl.inc());
        
        // clear container
        con.clear();
        
        std::vector<int> currShell;
        for (int i=0; i<ncore; i++)
        {
                currShell.clear();
                currShell.push_back(corelist[i]);

                for (int k=1; k <= maxshell; k++)
                {
                        std::vector<int> prevShell(currShell);
                        currShell.clear();
                        for (unsigned int j=0; j < prevShell.size(); j++)
                        {
                                for (std::set<int>::iterator it=neighbours[prevShell[j]].begin(); it != neighbours[prevShell[j]].end(); it++)
                                {
                                        if (ds[corelist[i]][*it]==-1)
                                        {
                                                ds[corelist[i]][*it]=k;
                                                currShell.push_back(*it);
                                        }
                                }
                        }
                }
        }
        neighbours.clear();
        currShell.clear();
        delete[] xyz;
        delete[] ds;
        delete[] face_areas;
}

void _calcTessellationVolSurfAtomic(double *xyz_ptr, float boxlength, int32_t *f2c, int natoms, int nmolecules, int maxshell, char *ds_ptr, int32_t *corelist, int ncore, float* vols, float* face_area_ptr)
{
//         std::cout << "_calcTessellation()" << std::endl;
        int nblocks=25;

	double **xyz = new double*[natoms];
	for (int i=0;i<natoms;i++) 
	  xyz[i] = &(xyz_ptr[i*3]);
	
        char **ds = new char*[nmolecules];
        for (int i=0; i<nmolecules; i++) {
            ds[i]=&(ds_ptr[i*nmolecules]);
	    for (int j=0; j<nmolecules; j++) {
                if (i==j) ds[i][j]=0;
                else ds[i][j]=-1;
            }
	}
        
        voro::container con(0,boxlength,0,boxlength,0,boxlength,nblocks,nblocks,nblocks,true,true,true,8);

        std::vector<std::set<int> > neighbours;
        
        neighbours.resize(nmolecules);
        
        // add particle coordinates to the container
        for (int i=0;i<natoms;i++) con.put(i,xyz[i][0],xyz[i][1],xyz[i][2]);
        
        // calculate tessellation
        voro::voronoicell_neighbor c;
        voro::c_loop_all vl(con);
        // neighbour list
        std::vector<int> nb;
	// face areas --- order is the same as in neighbor list
	std::vector<double> fa;

        if (vl.start()) do {
            con.compute_cell(c,vl); 
            c.neighbors(nb);
	    c.face_areas(fa);
	    vols[con.id[vl.ijk][vl.q]] += c.volume();
            for (unsigned int i=0;i<nb.size();i++) {
// 		    con.id[vl.ijk][vl.q]                 ID of current particle
// 		    nb[i] 				 current neighbor
// 		    insert neighbours
                neighbours[f2c[con.id[vl.ijk][vl.q]]].insert(f2c[nb[i]]);
		face_area_ptr[con.id[vl.ijk][vl.q]]+=fa[i];
            }
            nb.clear();
	    fa.clear();
        }
        while (vl.inc());
        
        // clear container
        con.clear();
        
        std::vector<int> currShell;
        for (int i=0; i<ncore; i++)
        {
                currShell.clear();
                currShell.push_back(corelist[i]);

                for (int k=1; k <= maxshell; k++)
                {
                        std::vector<int> prevShell(currShell);
                        currShell.clear();
                        for (unsigned int j=0; j < prevShell.size(); j++)
                        {
                                for (std::set<int>::iterator it=neighbours[prevShell[j]].begin(); it != neighbours[prevShell[j]].end(); it++)
                                {
                                        if (ds[corelist[i]][*it]==-1)
                                        {
                                                ds[corelist[i]][*it]=k;
                                                currShell.push_back(*it);
                                        }
                                }
                        }
                }
        }
        neighbours.clear();
        currShell.clear();
        delete[] xyz;
        delete[] ds;
}


void _calcTessellationParallel(double *xyz, int *f2c, int *corelist, int *surroundlist, char *delaunay_ts, float boxl, int natoms, int nmolecules, int maxshell, int ncore, int nsurr, int ncpu)
{

  double **xyz_ts = new double*[ncpu];
  for (int i(0); i<ncpu; i++) {
    xyz_ts[i] = &(xyz[i*natoms*3]);
  }

#pragma omp parallel num_threads(ncpu)
  {
    //int *ds = new int[nmolecules*nmolecules];
    char *ds = new char[nmolecules*nmolecules];
    int tid = omp_get_thread_num();
    char dist, mindist;
    int /*dist, mindist,*/ icore, isurr;

    _calcTessellation(xyz_ts[tid], boxl, f2c, natoms, nmolecules, maxshell, ds, corelist, ncore);

    for (int i(0); i<nsurr; i++) {
      mindist = maxshell+1;
      isurr = surroundlist[i];
      for (int j(0); j<ncore; j++) {
    	icore = corelist[j];
    	dist = ds[icore*nmolecules + isurr];
    	if (dist < mindist && dist != -1)
    	  mindist = dist;
      }
      delaunay_ts[tid*nsurr + i] = /*(char)*/ mindist;
    }
    
    delete[] ds;
  }

  delete[] xyz_ts;
}

void _calcTessellationParallelAll(double *xyz, int *f2c, int *corelist, char *delaunay_ts, float boxl, int natoms, int nmolecules, int maxshell, int ncore, int ncpu)
{

  double **xyz_ts = new double*[ncpu];
  for (int i(0); i<ncpu; i++) {
    xyz_ts[i] = &(xyz[i*natoms*3]);
  }

#pragma omp parallel num_threads(ncpu)
  {
    int tid = omp_get_thread_num();

    _calcTessellation(xyz_ts[tid], boxl, f2c, natoms, nmolecules, maxshell, &(delaunay_ts[tid*nmolecules*nmolecules]), corelist, ncore);
  }

  delete[] xyz_ts;
}

// compare example polygons.cc
void _drawTessellation(double *xyz_ptr, float box_x, float box_y, float box_z, int npoints, int *points_to_draw, int npoints_to_draw, double cylinder_radius, bool triangles, int nmol, int color_id, const char* filename)
{
        int nblocks=20;

	double x2 = double(box_x)/2.0;
	double y2 = double(box_y)/2.0;
	double z2 = double(box_z)/2.0;
	
	double **xyz = new double*[npoints];
	for (int i=0;i<npoints;i++) 
	  xyz[i] = &(xyz_ptr[i*3]);

        voro::container con(0,box_x,0,box_y,0,box_z,nblocks,nblocks,nblocks,true,true,true,8);

        // add particle coordinates to the container
        for (int i=0;i<npoints;i++) con.put(i,xyz[i][0]+x2,xyz[i][1]+y2,xyz[i][2]+z2);
        
        // calculate tessellation
        voro::voronoicell_neighbor c;
        voro::c_loop_all vl(con);

	if (vl.start()) do {
            con.compute_cell(c,vl); 
        }
        while (vl.inc());

	// tmp vertex list
	std::vector<double> vxtmp, ftmp;

	// vertices per face lists
	std::vector< std::vector<double> > vx1, vx2, vx3;

	// neighbor list
	std::vector<int> n;

	// vertices per face
	std::vector<int> f_vert;
	
	// open file for output
	FILE *fp = fopen(filename,"a+");
	
	int idx, k;
	unsigned int i, j, ii;
	bool toDraw;
	double x,y,z;

	if (vl.start()) do if (con.compute_cell(c,vl)) {
	      vl.pos(x,y,z);
	      c.vertices(x,y,z,vxtmp);
	      c.face_vertices(f_vert);
	      c.neighbors(n);
	      idx = vl.pid();

	      //if (idx == 0) std::cout << x-x2 << "\t" << y-y2 << "\t" << z-z2 << std::endl;
	      
	      toDraw = false;
	      for (k=0; k<npoints_to_draw; k++){
		if (points_to_draw[k] == idx) {
		  toDraw = true;
		  break;
		}
	      }

	      j = 0;
	      // loop over all faces
	      for (i=0; i<n.size(); i++) {
		ftmp.clear();		
		  int k, l, n=f_vert[j];
		  // loop over vertices contained in this face
		  for (k=0; k<n; k++) {
		    l=3*f_vert[j+k+1];
		    // add vertex x y z to list
		    ftmp.push_back(vxtmp[l]);
		    ftmp.push_back(vxtmp[l+1]);
		    ftmp.push_back(vxtmp[l+2]);
		  }
		  // append temp. list to core or surround face list
		  if (toDraw) vx1.push_back(ftmp);
		  else vx2.push_back(ftmp);
		// skip to next entry in face vertex list
		j += f_vert[j]+1;
	      }
	      
	    } while (vl.inc());

	// make intersection of face sets
	// loop over toDraw set (vx1)
	for (i=0; i<vx1.size(); i++) {
	    // loop over vx2 sets
	    for (j=0; j<vx2.size(); j++) {
	      // first, check if the faces have the same number of vertices
	      if (vx1[i].size() != vx2[j].size()) continue;

	      // if they have the same size, make copies of the vertex sets, sort and compare them
	      std::vector<double> set1 (vx1[i]); std::sort(set1.begin(), set1.end());
	      std::vector<double> set2 (vx2[j]); std::sort(set2.begin(), set2.end());
	      bool identical = true;

	      // compare all elements of sorted lists
	      for (ii=0; ii<set1.size(); ii++) {
		if (fabs(set1[ii] - set2[ii]) > 0.000001) {
		  identical = false;
		  break;
		}
	      }

	      // if the sets are identical, append to intersection list
	      if (identical) {
		vx3.push_back(vx1[i]);
	      }
	    }
	}
	
	fprintf(fp,"#0\n\nmol new\ngraphics %d color %d\n",nmol,color_id);
	if (cylinder_radius > double(0)) {
	  for (i=0; i<vx3.size(); i++) {
	    unsigned int size = vx3[i].size();
	    for (j=0; j<size-3; j+=3) {
	      fprintf(fp, "graphics %d cylinder { %g %g %g } { %g %g %g } radius %g\n", nmol, vx3[i][j]-x2, vx3[i][j+1]-y2, vx3[i][j+2]-z2, vx3[i][j+3]-x2, vx3[i][j+4]-y2, vx3[i][j+5]-z2, cylinder_radius);
	    }
	    fprintf(fp, "graphics %d cylinder { %g %g %g } { %g %g %g } radius %g\n", nmol, vx3[i][size-3]-x2, vx3[i][size-2]-y2, vx3[i][size-1]-z2, vx3[i][0]-x2, vx3[i][1]-y2, vx3[i][2]-z2, cylinder_radius);
	  }
	}

	if (triangles) {
	  for (i=0; i<vx3.size(); i++) {
	    unsigned int size = vx3[i].size();
	    for (j=3; j<size-3; j+=3) {
	      fprintf(fp, "graphics %d triangle { %g %g %g } { %g %g %g } { %g %g %g }\n", nmol, vx3[i][0]-x2, vx3[i][1]-y2, vx3[i][2]-z2, vx3[i][j]-x2, vx3[i][j+1]-y2, vx3[i][j+2]-z2, vx3[i][j+3]-x2, vx3[i][j+4]-y2, vx3[i][j+5]-z2);
	    }
	  }
	}
	
        // cleanup
        con.clear();
	fclose(fp);
	delete[] xyz;
}
