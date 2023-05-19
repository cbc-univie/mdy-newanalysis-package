void test_parallelism(int iterations);

void atomic_ind_dip(double *, double *, double *, double *, double *, double *, int, int, double);
void atomic_ind_dip_per_atom(int, double *, double *, double *, double *, double *, double *, int, int, double);
void derive_ind_dip(double *, double *, double *, double *, int, double);
void derive_ind_dip_per_atom(int, double *, double *, double *, double *, int, double);

void calc_accumulate_shellwise_g_function(double *, double *, char *, double *, double *, int, int, int, int, double, double,  double);
int get_best_index(double *, double *, int);
void write_Mr_diagram(double *, double *, int, double *, double *, double *, double, int, int);
void write_Kirkwood_diagram(double *, double *, double *, double *, int, double *, double, int, int);
void write_Kirkwood_diagram_shellwise(double *, double *, double *, double *, int, double *, char *, int, double, int, int);
void write_Kirkwood_diagram_2D(double *, double *, double *, double *, int, double *, double, int, int, int);

void calc_donor_grid(double *, int *, int, double);

void calc_sum_Venn_MD_Cage_Single(double *, double *, int, char *, char *, int, int);
void separateCollectiveDipolesSpherically(double *, double *, int, double *, double *, double *, double);

void correlateSingleVectorTS(double *, double *, int, int);
void crossCorrelateSingleVectorTS(double *, double *, double *, int, int, bool, int);
void correlateMultiVectorTS(double *, double *, int, int, int);
void correlateMultiVectorShellwiseTS(double *, double *, double *, int, int, int, int);
void correlateMultiVectorVennShellwiseTS(double *, double *,  double *, double *, int, int, int, int, int);
void crossCorrelateMultiVectorTS(double *, double *, double *, int , int, int, bool, int);

void calcVanHoveSingleVector(double *, double *, int, int);
void calcVanHoveMultiVector(double *, double *, int, int, int);

void sort_collective_dip_NN_shells(char *, double *, double *, int);
void sort_collective_dip_NN_shells_int(int *, double *, double *, int);

void calc_dip_ten_collective(double *, int, double *);
void calc_dip_ten_collective_per_atom(int, double *, int, double *);
void calc_dip_ten_collective_cross(double *, int, double *, int, double *);
void calc_dip_ten_collective_NNshellwise(double *, int, char *, int, double *);
void calc_dip_ten_collective_NNshellwise_self(double *, int *, int, int, char *, int, int, double *);
void calc_dip_ten_collective_NNshellwise_cross(double *, double *, int *, int, int, int, char *, int, int, int, double *);
void calc_dip_ten_collective_1Nshellwise_self(double *, int, char *, int, double *);
void calc_dip_ten_collective_1Nshellwise_cross(double *, int, double *, int, char *, int, double *);
void calc_dip_ten_collective_shellwise(double *, int, char *, int, double *);
void calc_dip_ten_collective_vennshellwise(double *, int, char *, char *, int, int, double *);

void calc_distance_delauny_mindist(int *, double *, int, double, int, double);
void calc_distance_delauny(int *, double *, int, int, double);
void calc_min_dist_tesselation(double *, double *, double *, int, int, double);

void construct_relay_matrix(double *, double *, double *, int);

void pairiter_loop_(int, int *, int *, double *, int, int, int *, int, double *, double *, double *, int *, int);
void dipten_double_loop_(double *, double *, double *, int, int, int);
void dipten_double_loop2_(double *, double *, double *, int, int, int);
