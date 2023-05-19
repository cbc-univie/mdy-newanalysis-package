void _calcTessellation(double *xyz_ptr, float boxlength, int32_t *f2c, int natoms, int nmolecules, int maxshell, char *ds_ptr, int32_t *corelist, int ncore);
void _calcTessellationVolSurf(double *xyz_ptr, float boxlength, int32_t *f2c, int natoms, int nmolecules, int maxshell, char *ds_ptr, int32_t *corelist, int ncore, float* vols, float* face_area_ptr);
void _drawTessellation(double *xyz_ptr, float box_x, float box_y, float box_z, int npoints, int *points_to_draw, int npoints_to_draw, double cylinder_radius, bool triangles, int nmol, int color_id, const char* filename);
void _calcTessellationParallel(double *xyz, int *f2c, int *corelist, int *surroundlist, char *delaunay_ts, float boxl, int natoms, int nmolecules, int maxshell, int ncore, int nsurr, int ncpu);
void _calcTessellationParallelAll(double *xyz, int *f2c, int *corelist, char *delaunay_ts, float boxl, int natoms, int nmolecules, int maxshell, int ncore, int ncpu);
void _calcTessellationVolSurfAtomic(double *xyz_ptr, float boxlength, int32_t *f2c, int natoms, int nmolecules, int maxshell, char *ds_ptr, int32_t *corelist, int ncore, float* vols, float* face_area_ptr);

