
#include <fstream>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <vector> 
//#include <random>

#include <bits/stdc++.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <Eigen/QR>
#include "time.h"

#include "genotype.h"
#include "mailman.h"
#include "arguments.h"
//#include "helper.h"
#include "storage.h"

#if SSE_SUPPORT==1
	#define fastmultiply fastmultiply_sse
	#define fastmultiply_pre fastmultiply_pre_sse
#else
	#define fastmultiply fastmultiply_normal
	#define fastmultiply_pre fastmultiply_pre_normal
#endif

using namespace Eigen;
using namespace std;

// Storing in RowMajor Form
typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixXdr;
//Intermediate Variables
int blocksize;
int hsegsize;
double *partialsums;
double *sum_op;		
double *yint_e;
double *yint_m;
double **y_e;
double **y_m;


struct timespec t0;

clock_t total_begin = clock();
MatrixXdr pheno;
MatrixXdr mask;
MatrixXdr covariate;  
MatrixXdr Q;
MatrixXdr v1; //W^ty
MatrixXdr v2;            //QW^ty
MatrixXdr v3;    //WQW^ty
MatrixXdr new_pheno;



genotype g;
genotype g1;
genotype g2;
MatrixXdr geno_matrix; //(p,n)
genotype* Geno;
int MAX_ITER;
int k,p,n;
int k_orig;

MatrixXdr c; //(p,k)
MatrixXdr x; //(k,n)
MatrixXdr v; //(p,k)
MatrixXdr means; //(p,1)
MatrixXdr stds; //(p,1)
MatrixXdr sum2;
MatrixXdr sum;  
////////
//related to phenotype	
double y_sum; 
double y_mean;

options command_line_opts;

bool debug = false;
bool check_accuracy = false;
bool var_normalize=false;
int accelerated_em=0;
double convergence_limit;
bool memory_efficient = false;
bool missing=false;
bool fast_mode = true;
bool text_version = false;
bool use_cov=false; 


//// jackknife index wich are computed based on annotation file
MatrixXdr dic_index;
MatrixXdr jack_bin_size;
vector<int> len;
vector<int> Annot;
int Njack=100;
int Nbin=8;
int Nz=10;
///////

//define random vector z's
MatrixXdr  all_zb;
MatrixXdr res;
MatrixXdr XXz;
MatrixXdr Xy;
MatrixXdr yXXy;






std::istream& newline(std::istream& in)
{
    if ((in >> std::ws).peek() != std::char_traits<char>::to_int_type('\n')) {
        in.setstate(std::ios_base::failbit);
    }
    return in.ignore();
}


int read_cov(bool std,int Nind, std::string filename, std::string covname){
	ifstream ifs(filename.c_str(), ios::in); 
	std::string line; 
	std::istringstream in; 
	int covIndex = 0; 
	std::getline(ifs,line); 
	in.str(line); 
	string b;
	vector<vector<int> > missing; 
	int covNum=0;  
	while(in>>b)
	{
		if(b!="FID" && b !="IID"){
		missing.push_back(vector<int>()); //push an empty row  
		if(b==covname && covname!="")
			covIndex=covNum; 
		covNum++; 
		}
	}
	vector<double> cov_sum(covNum, 0); 
	if(covname=="")
	{
		covariate.resize(Nind, covNum); 
		cout<< "Read in "<<covNum << " Covariates.. "<<endl;
	}
	else 
	{
		covariate.resize(Nind, 1); 
		cout<< "Read in covariate "<<covname<<endl;  
	}

	
	int j=0; 
	while(std::getline(ifs, line)){
		in.clear(); 
		in.str(line);
		string temp;
		in>>temp; in>>temp; //FID IID 
		for(int k=0; k<covNum; k++){
			
			in>>temp;
			if(temp=="NA")
			{
				missing[k].push_back(j);
				continue; 
			} 
			double cur = atof(temp.c_str()); 
			if(cur==-9)
			{
				missing[k].push_back(j); 
				continue; 
			}
			if(covname=="")
			{
				cov_sum[k]= cov_sum[k]+ cur; 
				covariate(j,k) = cur; 
			}
			else
				if(k==covIndex)
				{
					covariate(j, 0) = cur;
					cov_sum[k] = cov_sum[k]+cur; 
				}
		}
		//if(j<10) 
		//	cout<<covariate.block(j,0,1, covNum)<<endl; 
		j++;
	}
	//compute cov mean and impute 
	for (int a=0; a<covNum ; a++)
	{
		int missing_num = missing[a].size(); 
		cov_sum[a] = cov_sum[a] / (Nind - missing_num);

		for(int b=0; b<missing_num; b++)
		{
                        int index = missing[a][b];
                        if(covname=="")
                                covariate(index, a) = cov_sum[a];
                        else if (a==covIndex)
                                covariate(index, 0) = cov_sum[a];
                } 
	}
	if(std)
	{
		MatrixXdr cov_std;
		cov_std.resize(1,covNum);  
		MatrixXdr sum = covariate.colwise().sum();
		MatrixXdr sum2 = (covariate.cwiseProduct(covariate)).colwise().sum();
		MatrixXdr temp;
//		temp.resize(Nind, 1); 
//		for(int i=0; i<Nind; i++)
//			temp(i,0)=1;  
		for(int b=0; b<covNum; b++)
		{
			cov_std(0,b) = sum2(0,b) + Nind*cov_sum[b]*cov_sum[b]- 2*cov_sum[b]*sum(0,b);
			cov_std(0,b) =sqrt((Nind- 1)/cov_std(0,b)) ;
			double scalar=cov_std(0,b); 
			for(int j=0; j<Nind; j++)
			{
				covariate(j,b) = covariate(j,b)-cov_sum[b];  
				covariate(j,b) =covariate(j,b)*scalar;
			} 
			//covariate.col(b) = covariate.col(b) -temp*cov_sum[b];
			
		}
	}	
	return covNum; 
}

/*void read_cov(int Nind, std::string filename, std::string covname){
	ifstream ifs(filename.c_str(), ios::in); 
	std::string line; 
	std::istringstream in; 
	int covIndex = 0; 
	std::getline(ifs,line); 
	in.str(line); 
	string b;
	vector<vector<int> > missing; 
	int covNum=0;  
	while(in>>b)
	{
		missing.push_back(vector<int>()); //push an empty row  
		if(b==covname && covname!="")
			covIndex=covNum; 
		covNum++; 
	}
	vector<double> cov_sum(covNum, 0); 
	if(covname=="")
	{
		covariate.resize(Nind, covNum); 
		cout<< "Read in "<<covNum << " Covariates.. "<<endl;
	}
	else 
	{
		covariate.resize(Nind, 1); 
		cout<< "Read in covariate "<<covname<<endl;  
	}

	
	int j=0; 
	while(std::getline(ifs, line)){
		in.clear(); 
		in.str(line);
		string temp; 
		for(int k=0; k<covNum; k++){
			in>>temp;
			if(temp=="NA")
			{
				missing[k].push_back(j);
				continue;  
			} 
			int cur = atof(temp.c_str()); 
			if(cur==-9)
			{
				missing[k].push_back(j); 
				continue; 
			}
			if(covname=="")
			{
				cov_sum[k]= cov_sum[k]+ cur; 
				covariate(j,k) = cur; 
			}
			else
				if(k==covIndex)
				{
					covariate(j, 0) = cur;
					cov_sum[k] = cov_sum[k]+cur; 
				}
		} 
		j++;
	}
	//compute cov mean and impute 
	for (int a=0; a<covNum ; a++)
	{
		int missing_num = missing[a].size(); 
		cov_sum[a] = cov_sum[a] / (covNum - missing_num);

		for(int b=0; b<missing_num; b++)
		{
                        int index = missing[a][b];
                        if(covname=="")
                                covariate(index, a) = cov_sum[a];
                        else if (a==covIndex)
                                covariate(index, 0) = cov_sum[a];
                } 
	}
}*/
void read_pheno2(int Nind, std::string filename){
//	pheno.resize(Nind,1); 
	ifstream ifs(filename.c_str(), ios::in); 
	
	std::string line;
	std::istringstream in;  
	int phenocount=0; 
//read header
	std::getline(ifs,line); 
	in.str(line); 
	string b; 
	while(in>>b)
	{
		if(b!="FID" && b !="IID")
			phenocount++; 
	}
	pheno.resize(Nind, phenocount);
	mask.resize(Nind, phenocount);
	int i=0;  
	while(std::getline(ifs, line)){
		in.clear(); 
		in.str(line); 
		string temp;
		//fid,iid
		//todo: fid iid mapping; 
		//todo: handle missing phenotype
		in>>temp; in>>temp; 
		for(int j=0; j<phenocount;j++) {
			in>>temp;
			double cur = atof(temp.c_str());
			if(temp=="NA" || cur==-9){
			pheno(i,j)=0;
			mask(i,j)=0;
			}
			else{
			pheno(i,j)=atof(temp.c_str());
			mask(i,j)=1;

			}

    
		}
		i++;
	}
	//cout<<pheno; 
}
void read_pheno(int Nind, std::string filename){
	pheno.resize(Nind, 1); 
	ifstream ifs(filename.c_str(), ios::in); 
	
	std::string line;
	int i=0;  
	while(std::getline(ifs, line)){
		pheno(i,0) = atof(line.c_str());
		if(pheno(i,0)==-1)
			cout<<"WARNING: missing phenotype"<<endl; 
		i++;  
	}

}
void multiply_y_pre_fast(MatrixXdr &op, int Ncol_op ,MatrixXdr &res,bool subtract_means){
	
	for(int k_iter=0;k_iter<Ncol_op;k_iter++){
		sum_op[k_iter]=op.col(k_iter).sum();		
	}

			//cout << "Nops = " << Ncol_op << "\t" <<g.Nsegments_hori << endl;
	#if DEBUG==1
		if(debug){
			print_time (); 
			cout <<"Starting mailman on premultiply"<<endl;
			cout << "Nops = " << Ncol_op << "\t" <<g.Nsegments_hori << endl;
			cout << "Segment size = " << g.segment_size_hori << endl;
			cout << "Matrix size = " <<g.segment_size_hori<<"\t" <<g.Nindv << endl;
			cout << "op = " <<  op.rows () << "\t" << op.cols () << endl;
		}
	#endif


	//TODO: Memory Effecient SSE FastMultipy

	for(int seg_iter=0;seg_iter<g.Nsegments_hori-1;seg_iter++){
		mailman::fastmultiply(g.segment_size_hori,g.Nindv,Ncol_op,g.p[seg_iter],op,yint_m,partialsums,y_m);
		int p_base = seg_iter*g.segment_size_hori; 
		for(int p_iter=p_base; (p_iter<p_base+g.segment_size_hori) && (p_iter<g.Nsnp) ; p_iter++ ){
			for(int k_iter=0;k_iter<Ncol_op;k_iter++) 
				res(p_iter,k_iter) = y_m[p_iter-p_base][k_iter];
		}
	}

	int last_seg_size = (g.Nsnp%g.segment_size_hori !=0 ) ? g.Nsnp%g.segment_size_hori : g.segment_size_hori;
	mailman::fastmultiply(last_seg_size,g.Nindv,Ncol_op,g.p[g.Nsegments_hori-1],op,yint_m,partialsums,y_m);		
	int p_base = (g.Nsegments_hori-1)*g.segment_size_hori;
	for(int p_iter=p_base; (p_iter<p_base+g.segment_size_hori) && (p_iter<g.Nsnp) ; p_iter++){
		for(int k_iter=0;k_iter<Ncol_op;k_iter++) 
			res(p_iter,k_iter) = y_m[p_iter-p_base][k_iter];
	}

	#if DEBUG==1
		if(debug){
			print_time (); 
			cout <<"Ending mailman on premultiply"<<endl;
		}
	#endif


	if(!subtract_means)
		return;

	for(int p_iter=0;p_iter<p;p_iter++){
 		for(int k_iter=0;k_iter<Ncol_op;k_iter++){		 
			res(p_iter,k_iter) = res(p_iter,k_iter) - (g.get_col_mean(p_iter)*sum_op[k_iter]);
			if(var_normalize)
				res(p_iter,k_iter) = res(p_iter,k_iter)/(g.get_col_std(p_iter));		
 		}		
 	}	

}

void multiply_y_post_fast(MatrixXdr &op_orig, int Nrows_op, MatrixXdr &res,bool subtract_means){

	MatrixXdr op;
	op = op_orig.transpose();

	if(var_normalize && subtract_means){
		for(int p_iter=0;p_iter<p;p_iter++){
			for(int k_iter=0;k_iter<Nrows_op;k_iter++)		
				op(p_iter,k_iter) = op(p_iter,k_iter) / (g.get_col_std(p_iter));		
		}		
	}

	#if DEBUG==1
		if(debug){
			print_time (); 
			cout <<"Starting mailman on postmultiply"<<endl;
		}
	#endif
	
	int Ncol_op = Nrows_op;

	//cout << "ncol_op = " << Ncol_op << endl;

	int seg_iter;
	for(seg_iter=0;seg_iter<g.Nsegments_hori-1;seg_iter++){
mailman::fastmultiply_pre(g.segment_size_hori,g.Nindv,Ncol_op, seg_iter * g.segment_size_hori, g.p[seg_iter],op,yint_e,partialsums,y_e);
	}
	int last_seg_size = (g.Nsnp%g.segment_size_hori !=0 ) ? g.Nsnp%g.segment_size_hori : g.segment_size_hori;
	mailman::fastmultiply_pre(last_seg_size,g.Nindv,Ncol_op, seg_iter * g.segment_size_hori, g.p[seg_iter],op,yint_e,partialsums,y_e);

	for(int n_iter=0; n_iter<n; n_iter++)  {
		for(int k_iter=0;k_iter<Ncol_op;k_iter++) {
			res(k_iter,n_iter) = y_e[n_iter][k_iter];
			y_e[n_iter][k_iter] = 0;
		}
	}
	
	#if DEBUG==1
		if(debug){
			print_time (); 
			cout <<"Ending mailman on postmultiply"<<endl;
		}
	#endif


	if(!subtract_means)
		return;

	double *sums_elements = new double[Ncol_op];
 	memset (sums_elements, 0, Nrows_op * sizeof(int));

 	for(int k_iter=0;k_iter<Ncol_op;k_iter++){		
 		double sum_to_calc=0.0;		
 		for(int p_iter=0;p_iter<p;p_iter++)		
 			sum_to_calc += g.get_col_mean(p_iter)*op(p_iter,k_iter);		
 		sums_elements[k_iter] = sum_to_calc;		
 	}		
 	for(int k_iter=0;k_iter<Ncol_op;k_iter++){		
 		for(int n_iter=0;n_iter<n;n_iter++)		
 			res(k_iter,n_iter) = res(k_iter,n_iter) - sums_elements[k_iter];		
 	}


}

void multiply_y_pre_naive_mem(MatrixXdr &op, int Ncol_op ,MatrixXdr &res){
	for(int p_iter=0;p_iter<p;p_iter++){
		for(int k_iter=0;k_iter<Ncol_op;k_iter++){
			double temp=0;
			for(int n_iter=0;n_iter<n;n_iter++)
				temp+= g.get_geno(p_iter,n_iter,var_normalize)*op(n_iter,k_iter);
			res(p_iter,k_iter)=temp;
		}
	}
}

void multiply_y_post_naive_mem(MatrixXdr &op, int Nrows_op ,MatrixXdr &res){
	for(int n_iter=0;n_iter<n;n_iter++){
		for(int k_iter=0;k_iter<Nrows_op;k_iter++){
			double temp=0;
			for(int p_iter=0;p_iter<p;p_iter++)
				temp+= op(k_iter,p_iter)*(g.get_geno(p_iter,n_iter,var_normalize));
			res(k_iter,n_iter)=temp;
		}
	}
}

void multiply_y_pre_naive(MatrixXdr &op, int Ncol_op ,MatrixXdr &res){
	res = geno_matrix * op;
}

void multiply_y_post_naive(MatrixXdr &op, int Nrows_op ,MatrixXdr &res){
	res = op * geno_matrix;
}

void multiply_y_post(MatrixXdr &op, int Nrows_op ,MatrixXdr &res,bool subtract_means){
    if(fast_mode)
        multiply_y_post_fast(op,Nrows_op,res,subtract_means);
    else{
		if(memory_efficient)
			multiply_y_post_naive_mem(op,Nrows_op,res);
		else
			multiply_y_post_naive(op,Nrows_op,res);
	}
}

void multiply_y_pre(MatrixXdr &op, int Ncol_op ,MatrixXdr &res,bool subtract_means){
    if(fast_mode)
        multiply_y_pre_fast(op,Ncol_op,res,subtract_means);
    else{
		if(memory_efficient)
			multiply_y_pre_naive_mem(op,Ncol_op,res);
		else
			multiply_y_pre_naive(op,Ncol_op,res);
	}
}

void initial_var(int key)
{
    /*if(key==1)
        g=g1;
    if(key==2)
    	g=g2;*/
   // g=Geno[key];
    p = g.Nsnp;
	n = g.Nindv;


	c.resize(p,k);
	x.resize(k,n);
	v.resize(p,k);
	means.resize(p,1);
	stds.resize(p,1);
	sum2.resize(p,1); 
	sum.resize(p,1); 

	if(!fast_mode && !memory_efficient){
		geno_matrix.resize(p,n);
		g.generate_eigen_geno(geno_matrix,var_normalize);
	}

	//TODO: Initialization of c with gaussian distribution
	c = MatrixXdr::Random(p,k);


	// Initial intermediate data structures
	blocksize = k;
	 hsegsize = g.segment_size_hori; 	// = log_3(n)
	int hsize = pow(3,hsegsize);		 
	int vsegsize = g.segment_size_ver; 		// = log_3(p)
	int vsize = pow(3,vsegsize);		 

	partialsums = new double [blocksize];
	sum_op = new double[blocksize];
	yint_e = new double [hsize*blocksize];
	yint_m = new double [hsize*blocksize];
	memset (yint_m, 0, hsize*blocksize * sizeof(double));
	memset (yint_e, 0, hsize*blocksize * sizeof(double));

	y_e  = new double*[g.Nindv];
	for (int i = 0 ; i < g.Nindv ; i++) {
		y_e[i] = new double[blocksize];
		memset (y_e[i], 0, blocksize * sizeof(double));
	}

	y_m = new double*[hsegsize];
	for (int i = 0 ; i < hsegsize ; i++)
		y_m[i] = new double[blocksize];
	for(int i=0;i<p;i++){
		means(i,0) = g.get_col_mean(i);
		stds(i,0) =1/g.get_col_std(i);
		//sum2(i,0) =g.get_col_sum2(i); 
		sum(i,0)= g.get_col_sum(i); 
	}




}

MatrixXdr multi_Xz (MatrixXdr zb){

              for(int j=0; j<g.Nsnp;j++)
                zb(j,0) =zb(j,0) *stds(j,0);
                                              
		MatrixXdr new_zb = zb.transpose(); 
	        MatrixXdr new_res(1, g.Nindv);
		multiply_y_post_fast(new_zb, 1, new_res, false); 
		MatrixXdr new_resid(1, g.Nsnp); 
		MatrixXdr zb_scale_sum = new_zb * means;
		new_resid = zb_scale_sum * MatrixXdr::Constant(1,g.Nindv, 1);
		new_res=new_res - new_resid;
       return new_res;

}	

double compute_yVKVy(int s){
	MatrixXdr new_pheno_sum = new_pheno.colwise().sum();
	MatrixXdr res(g.Nsnp, 1); 
	multiply_y_pre_fast(new_pheno,1,res,false); 
	res = res.cwiseProduct(stds); 
	MatrixXdr resid(g.Nsnp, 1); 
	resid = means.cwiseProduct(stds); 
	resid = resid *new_pheno_sum; 	
	MatrixXdr Xy(g.Nsnp,1); 
	Xy = res-resid; 
	double ytVKVy = (Xy.array()* Xy.array()).sum(); 
	ytVKVy = ytVKVy/g.Nsnp; 
	return ytVKVy;

}

double compute_yXXy(){

//for (int i=0;i<g.Nindv;i++)
	//pheno(i,0)=1;

        MatrixXdr res(g.Nsnp, 1);
        multiply_y_pre_fast(pheno,1,res,false);
        res = res.cwiseProduct(stds);
        MatrixXdr resid(g.Nsnp, 1);
        resid = means.cwiseProduct(stds);
        resid = resid *y_sum;
        MatrixXdr Xy(g.Nsnp,1);
        Xy = res-resid;
    
        double yXXy = (Xy.array()* Xy.array()).sum();
        

        return yXXy;

}





 double compute_tr_k(int s){
	  
	   /* if (s==1)
         initial_var(1);  
        if(s==2) 
         initial_var(2); 
*/
        initial_var(s);
	    double tr_k =0 ;
 		MatrixXdr temp = sum2 + g.Nindv* means.cwiseProduct(means) - 2 * means.cwiseProduct(sum);
		temp = temp.cwiseProduct(stds);
		temp = temp.cwiseProduct(stds); 
		tr_k = temp.sum() / g.Nsnp;
	   
	//    cout<<g.Nindv<<"    "<<g.Nsnp<<" s:   "<<temp.sum()<<"\n"; 
	//    cout<<tr_k<<"\n";
	   return tr_k;
	}
	
MatrixXdr  compute_XXz (){

	//mask
	for (int i=0;i<Nz;i++)
	   for(int j=0;j<g.Nindv;j++)
		 all_zb(j,i)=all_zb(j,i)*mask(j,0);

         res.resize(g.Nsnp, Nz);
         multiply_y_pre_fast(all_zb,Nz,res, false);

//         cout<<res<<endl;

        MatrixXdr zb_sum = all_zb.colwise().sum();
        

	for(int j=0; j<g.Nsnp; j++)
            for(int k=0; k<Nz;k++)
                 res(j,k) = res(j,k)*stds(j,0);

        MatrixXdr resid(g.Nsnp, Nz);
        MatrixXdr inter = means.cwiseProduct(stds);
        resid = inter * zb_sum;
        MatrixXdr inter_zb = res - resid;
       

	for(int k=0; k<Nz; k++)
            for(int j=0; j<g.Nsnp;j++)
                inter_zb(j,k) =inter_zb(j,k) *stds(j,0);

       MatrixXdr new_zb = inter_zb.transpose();
       MatrixXdr new_res(Nz, g.Nindv);
       
       multiply_y_post_fast(new_zb, Nz, new_res, false);
       
       MatrixXdr new_resid(Nz, g.Nsnp);
       MatrixXdr zb_scale_sum = new_zb * means;
       new_resid = zb_scale_sum * MatrixXdr::Constant(1,g.Nindv, 1);

	//return new_res;

                      /// new zb 
       MatrixXdr temp=new_res - new_resid;

	for (int i=0;i<Nz;i++)
           for(int j=0;j<g.Nindv;j++)
                 temp(i,j)=temp(i,j)*mask(j,0);


	return temp.transpose();
       

}


void read_annot (string filename){
        ifstream inp(filename.c_str());
        if (!inp.is_open()){
                cerr << "Error reading file "<< filename <<endl;
                exit(1);
        }
        string line;
        int j = 0 ;
        int linenum = 0 ;
        int num_parti;
        stringstream check1(line);
        string intermediate;
        vector <string> tokens;
        while(std::getline (inp, line)){
                linenum ++;
                char c = line[0];
                if (c=='#')
                        continue;
                istringstream ss (line);
                if (line.empty())
                        continue;
                j++;
                //cout<<line<<endl;

                stringstream check1(line);
                string intermediate;
                vector <string> tokens;
                // Tokenizing w.r.t. space ' ' 
                while(getline(check1, intermediate, ' '))
                 {
                      tokens.push_back(intermediate);
                 }
                 if(linenum==1){
                 num_parti=tokens.size();
                 if(num_parti!=Nbin)
                        cout<<"number of col of annot file does not match number of bins"<<endl;
                len.resize(num_parti,0);
                }
                int index_annot=0;
                for(int i = 0; i < tokens.size(); i++){
                        if (tokens[i]=="1")
                            index_annot=i;
                }
                   Annot.push_back(index_annot);
                   len[index_annot]++;
       }


	dic_index=MatrixXdr::Zero(Njack,Nbin);
  	jack_bin_size=MatrixXdr::Zero(Njack,Nbin);      


	int num_snps=0;
        for (int i=0;i<num_parti;i++){
                //cout<<len[i]<<endl;
                num_snps+=len[i];
        }
  //      cout<<"step size: "<<endl;
        int step_size=num_snps/Njack;
         int step_size_rem=num_snps%Njack;
    //    cout<<step_size<<endl;
      //  cout<<"reminder: "<<step_size_rem<<endl;
         int temp=step_size;

         //for (int i=0;i<Njack;i++)
           //      for (int j=0;j<num_parti;j++)
             //           dic_index(i,j)=0;

        j=1;

        for (int i=0;i<Annot.size();i++){
                if(i==(step_size*j) && j<Njack ){
                     j++;
                     for (int k=0; k<num_parti;k++)
                        dic_index(j-1,k)=dic_index(j-2,k);
                }
                dic_index(j-1,Annot[i])=dic_index(j-1,Annot[i])+1;
        }

        //handle not removing a bin in jackknife se
        for(int i=0;i<Nbin;i++){
	   for(int j=0;j<Njack;j++){
		if(j==0 && dic_index(j,i)==len[i]){
			dic_index(j,i)=len[i]/2;
			dic_index(j+1,i)=len[i]-dic_index(j,i);
		}
		else if ( j!=0 && (dic_index(j,i)-dic_index(j-1,i))==len[i] ){
			dic_index(j-1,i)=len[i]/2;
			dic_index(j,i)=len[i]-dic_index(j-1,i);
		}
	   }
       }
//cout<<"end reading annot"<<endl;

}

void count_fam(std::string filename){
        ifstream ifs(filename.c_str(), ios::in);

        std::string line;
        int i=0;
        while(std::getline(ifs, line)){
                i++;
        }
        g.Nindv=i-1;
}


int main(int argc, char const *argv[]){
  

parse_args(argc,argv);
////////////////////////////////////////////
///////////////////////////////////////////
    
    //MAX_ITER =  command_line_opts.max_iterations ; 
        int B = command_line_opts.batchNum;
        k_orig = command_line_opts.num_of_evec ;
        debug = command_line_opts.debugmode ;
        check_accuracy = command_line_opts.getaccuracy;
        var_normalize = false;
        accelerated_em = command_line_opts.accelerated_em;
        k = k_orig + command_line_opts.l;
        k = (int)ceil(k/10.0)*10;
        command_line_opts.l = k - k_orig;
        //p = g.Nsnp;
        //n = g.Nindv;
        bool toStop=false;
        toStop=true;
        srand((unsigned int) time(0));
        //Nz=10;
	Nz=command_line_opts.num_of_evec;
        k=Nz;
         ///clock_t io_end = clock();

	Njack=command_line_opts.jack_number;

////
string filename;
//filename="/home/alipazoki/filter4_no_mhc/mafld/annot.txt";
//filename="/home/alipazoki/RHEmc_online/example/annot_filter4.txt";
//filename=command_line_opts.Annot_PATH;
//read_annot(filename);	
//cout<<dic_index<<endl;
//cout<<jack_bin_size<<endl;
//////////////////////////// Read multi genotypes
string line;
int cov_num;
int num_files=0;
//string name="/home/alipazoki/filter4_no_mhc/mafld/adr.txt";
//string name="/home/alipazoki/UKBB/maf_ld/sub_indv/adr.txt";
string name=command_line_opts.GENOTYPE_FILE_PATH;
//cout<<name<<endl;
ifstream f (name.c_str());
while(getline(f,line))
   num_files++;    
   
string file_names[num_files];

int i=0;
ifstream ff (name.c_str());
while(getline(ff,line)) {
    
    file_names[i]=line;
    cout<<file_names[i]<<"\n";
    i++;
}

Nbin=num_files;
    
 cout<<"Number of the bins: "<<Nbin<<endl;   

filename=command_line_opts.Annot_PATH;
read_annot(filename);

///reading phnotype and save the number of indvs
//filename="/home/alipazoki/UKBB/kathy_pheno/height.pheno";    
//filename="/home/alipazoki/UKBB/maf_ld/sub_indv/10k.bmi.pheno";
filename=command_line_opts.PHENOTYPE_FILE_PATH;
count_fam(filename);
read_pheno2(g.Nindv,filename);
cout<<"Number of Indvs :"<<g.Nindv<<endl;
y_sum=pheno.sum();

//read covariate
//std::string covfile="/home/alipazoki/UKBB/kathy_pheno/height.covar";
//:qstd::string covfile="/home/alipazoki/UKBB/maf_ld/sub_indv/10k.bmi.covar";
//bool usee_cov=false;
//if(usee_cov==true){
std::string covfile=command_line_opts.COVARIATE_FILE_PATH;
std::string covname="";
if(covfile!=""){
     use_cov=true;
     cov_num=read_cov(false,g.Nindv, covfile, covname);
	//cout<<cov_num<<endl;
}
else if(covfile=="")
     cout<<"No Covariate File Specified"<<endl;

/// regress out cov from phenotypes

if(use_cov==true){
MatrixXdr mat_mask=mask.replicate(1,cov_num);
covariate=covariate.cwiseProduct(mat_mask);

MatrixXdr WtW= covariate.transpose()*covariate;
Q=WtW.inverse(); // Q=(W^tW)^-1
//cout<<" Number of covariates"<<cov_num<<endl;

MatrixXdr v1=covariate.transpose()*pheno; //W^ty
MatrixXdr v2=Q*v1;            //QW^ty
MatrixXdr v3=covariate*v2;    //WQW^ty
pheno=pheno-v3;
pheno=pheno.cwiseProduct(mask);
 }                 
////// normalize phenotype

//bool pheno_norm=false;
y_sum=pheno.sum();
y_mean = y_sum/mask.sum();

//if(pheno_norm==true){
for(int i=0; i<g.Nindv; i++){
   if(pheno(i,0)!=0)
      pheno(i,0) =pheno(i,0) - y_mean; //center phenotype
}
y_sum=pheno.sum();

//}








//define random vector z's
//Nz=1;

all_zb= MatrixXdr::Random(g.Nindv,Nz);
all_zb = all_zb * sqrt(3);
MatrixXdr output;
//define 

//e
//Njack=1;

XXz=MatrixXdr::Zero(g.Nindv,Nbin*(Njack+1)*Nz);
yXXy=MatrixXdr::Zero(Nbin,Njack+1);

for(int bin_index=0; bin_index<Nbin; bin_index++){

    std::stringstream f3;
    f3 << file_names[bin_index] << ".bed";
    string name=f3.str();
    cout<<name<<endl;
    ifstream ifs (name.c_str(), ios::in|ios::binary);
	
    g.read_header=true;
     //E
     for (int jack_index=0;jack_index<Njack;jack_index++){
	
	//cout<<"reading "<<jack_index<<"-th jckknf blck of "<<bin_index<<"-th bin"<<endl;
	
	if (jack_index==0){
		g.Nsnp=dic_index(jack_index,bin_index);
		jack_bin_size(jack_index,bin_index)=g.Nsnp;
	}
	else{
	        g.Nsnp=dic_index(jack_index,bin_index)-dic_index(jack_index-1,bin_index);
		jack_bin_size(jack_index,bin_index)=g.Nsnp;
        }
       if(g.Nsnp!=0){
	  //cout<<"Zero SNPSsss"<<endl;
       
	
	//g.Nsnp=len[bin_index];  
  	//cout<<"#SNPs"<<g.Nsnp<<endl; 
   	
	  g.read_plink(ifs,file_names[bin_index],missing,fast_mode);
    	  initial_var(0);
//////// do computation for i-th jack block of j-th bin 

	/// compute XXz
	output=compute_XXz();
//	cout<<output.col(0).sum()<<endl;
	for (int z_index=0;z_index<Nz;z_index++){
		 XXz.col((bin_index*(Njack+1)*Nz)+(jack_index*Nz)+z_index)=output.col(z_index);
		 XXz.col((bin_index*(Njack+1)*Nz)+(Njack*Nz)+z_index)+=output.col(z_index);   /// save whole sample
	}
	///compute yXXy
	yXXy(bin_index,jack_index)= compute_yXXy();
	yXXy(bin_index,Njack)+= yXXy(bin_index,jack_index);
	//// contribtion of each jackknife SUBSAMPLE (not block)
	
	


///////end computation
/////////////////////////////////destruct class g
	delete[] sum_op;
        delete[] partialsums;
        delete[] yint_e; 
        delete[] yint_m;
        for (int i  = 0 ; i < hsegsize; i++)
                delete[] y_m [i]; 
        delete[] y_m;

        for (int i  = 0 ; i < g.Nindv; i++)
                delete[] y_e[i]; 
        delete[] y_e;
	
	std::vector< std::vector<int> >().swap(g.p);
        std::vector< std::vector<int> >().swap(g.not_O_j);
        std::vector< std::vector<int> >().swap(g.not_O_i);
	
	//g.p.clear();
	//g.not_O_j.clear();
	//g.not_O_i.clear();
	g.columnsum.clear();
	g.columnsum2.clear();
	g.columnmeans.clear();
	g.columnmeans2.clear();
	//std::vector< std::vector<int> >().swap(g.columnsum);
	//std::vector< std::vector<int> >().swap(g.columnsum2);
	//std::vector< std::vector<double> >().swap(g.columnmeans);
	//std::vector< std::vector<double> >().swap(g.columnmeans2);
	g.read_header=false;
/////////////////////////////////////////////
       } //end of else 
   }// end of loop over jack blocks
          
  if(bin_index==0){
        //cout<<"zzzzzzzzzzzz"<<XXz.col((bin_index*Njack*Nz)+(Njack*Nz)).sum()<<endl;
        //cout<<bin_index<<" "<<Njack<<" "<<0<<endl; 
	//cout<<"zzzzzzzzzzzz"<<XXz.col((bin_index*Njack*Nz)+(Njack*Nz)).sum()<<endl;
	}
   // contribtion of each jackknife SUBSAMPLE (not block)
   
   
 for(int jack_index=0;jack_index<Njack;jack_index++){
	for (int z_index=0;z_index<Nz;z_index++){
	 MatrixXdr v1=XXz.col((bin_index*(Njack+1)*Nz)+(Njack*Nz)+z_index);
	// cout<<bin_index<<" "<<Njack<<" "<<z_index<<endl;

	//cout<<"v1: "<<v1.sum()<<endl; 
	MatrixXdr v2=XXz.col((bin_index*(Njack+1)*Nz)+(jack_index*Nz)+z_index); 
        // cout<<bin_index<<" "<<jack_index<<" "<<z_index<<endl;
	//cout<<"v2: "<<v2.sum()<<endl; 
	XXz.col((bin_index*(Njack+1)*Nz)+(jack_index*Nz)+z_index)=v1-v2;                    
        
	//cout<<"real v1: "<<XXz.col((bin_index*Njack*Nz)+(Njack*Nz)+z_index).sum()<<endl;
	//cout<<"real v2: "<<XXz.col((bin_index*Njack*Nz)+(jack_index*Nz)+z_index).sum()<<endl;

       }
	yXXy(bin_index,jack_index)=yXXy(bin_index,Njack)-yXXy(bin_index,jack_index);
}
 
} //end of loop over bins


/// normal equations LHS
MatrixXdr  A_trs(Nbin,Nbin);
MatrixXdr b_trk(Nbin,1);
MatrixXdr c_yky(Nbin,1);

MatrixXdr X_l(Nbin+1,Nbin+1);
MatrixXdr Y_r(Nbin+1,1);
//int bin_index=0;
int jack_index=Njack; 
MatrixXdr B1;
MatrixXdr B2;
MatrixXdr C1;
MatrixXdr C2;
double trkij;
double yy=(pheno.array() * pheno.array()).sum();
int Nindv_mask=mask.sum();
MatrixXdr jack;
MatrixXdr point_est;
MatrixXdr enrich_jack;
MatrixXdr enrich_point_est;

jack.resize(Nbin+1,Njack);
point_est.resize(Nbin+1,1);

enrich_jack.resize(Nbin,Njack);
enrich_point_est.resize(Nbin,1);


for (jack_index=0;jack_index<=Njack;jack_index++){

  for (int i=0;i<Nbin;i++){

	b_trk(i,0)=Nindv_mask;
	
	if(jack_index==Njack)
	c_yky(i,0)=yXXy(i,jack_index)/len[i];
	else
	c_yky(i,0)=yXXy(i,jack_index)/(len[i]-jack_bin_size(jack_index,i));
	//cout<<"bin "<<i<<"yXXy "<<yXXy(i,jack_index)<<endl;
	for (int j=i;j<Nbin;j++){
		//cout<<Njack<<endl;
		B1=XXz.block(0,(i*(Njack+1)*Nz)+(jack_index*Nz),g.Nindv,Nz);
		B2=XXz.block(0,(j*(Njack+1)*Nz)+(jack_index*Nz),g.Nindv,Nz);
		C1=B1.array()*B2.array();	
		C2=C1.colwise().sum();
		trkij=C2.sum();
		
		//cout<<"tr"<<i<<" "<<j<<" : "<<trkij<<endl;
	        if(jack_index==Njack)
		trkij=trkij/len[i]/len[j]/Nz;
		else
		 trkij=trkij/(len[i]-jack_bin_size(jack_index,i))/(len[j]-jack_bin_size(jack_index,j))/Nz;
		A_trs(i,j)=trkij;
		A_trs(j,i)=trkij;
		
	}
  }


X_l<<A_trs,b_trk,b_trk.transpose(),Nindv_mask;
Y_r<<c_yky,yy;

MatrixXdr herit=X_l.colPivHouseholderQr().solve(Y_r);

double temp_sig=0;
double temp_sum=0;

if(jack_index==Njack){
     for(int i=0;i<(Nbin+1);i++)
	  point_est(i,0)=herit(i,0);
}
else{
for(int i=0;i<(Nbin+1);i++)
      jack(i,jack_index)=herit(i,0);	
}

double total_val=0;
for(int i=0; i<Nbin;i++)
    total_val+=herit(i,0);

/*
for(int i=0; i<Nbin;i++){
   cout<<herit(i,0)/herit.sum()<<" ";
}
	cout<<total_val/herit.sum()<<endl;
*/
//cout<<X_l<<endl;
//cout<<Y_r<<endl;
//cout<<"ddddddddddd"<<endl;

}// end of loop over jacks

//cout<<"helolll"<<endl;
double temp_sig=0;
double temp_sum=0;

temp_sig=0;
temp_sum=point_est.sum();
for (int j=0;j<Nbin;j++){
        point_est(j,0)=point_est(j,0)/temp_sum;
        temp_sig+=point_est(j,0);
}
point_est(Nbin,0)=temp_sig;


for (int i=0;i<Njack;i++){
   temp_sig=0;
   temp_sum=jack.col(i).sum();
   for (int j=0;j<Nbin;j++){
        jack(j,i)=jack(j,i)/temp_sum;
        temp_sig+=jack(j,i);
   }
   jack(Nbin,i)=temp_sig;
}

///compute enrichment

double per_her;
double per_size;
int total_size=0;

for (int i=0;i<Nbin;i++)
   total_size+=len[i];

for (int j=0;j<Nbin;j++){
        per_her=point_est(j,0)/point_est(Nbin,0);
        per_size=(double)len[j]/total_size;
        enrich_point_est(j,0)=per_her/per_size;
	//cout<<j<<" "<<per_her<<" "<<total_size<<" "<<len[j]<<" "<<per_size<<" "<<enrich_point_est(j,0)<<endl;
}


for (int i=0;i<Njack;i++){
    per_size=0;
    total_size=0;
    for (int j=0;j<Nbin;j++){
	total_size+=(len[j]-jack_bin_size(i,j));
    }
   for (int j=0;j<Nbin;j++){
   	per_her=jack(j,i)/jack(Nbin,i);
	per_size=(double)(len[j]-jack_bin_size(i,j))/total_size;
	enrich_jack(j,i)=per_her/per_size;
	}
}



////compute jackknife SE
MatrixXdr sum_row=jack.rowwise().mean();
MatrixXdr SEjack;
SEjack=MatrixXdr::Zero(Nbin+1,1);
double temp_val=0;
for (int i=0;i<=Nbin;i++){
    for (int j=0;j<Njack;j++){
	temp_val=jack(i,j)-sum_row(i);
	temp_val= temp_val* temp_val;
	SEjack(i,0)+=temp_val;
    }
    SEjack(i,0)=SEjack(i,0)*(Njack-1)/Njack;
    SEjack(i,0)=sqrt(SEjack(i,0));
}
///////// compute jackknife SE of enrichment
 sum_row=enrich_jack.rowwise().mean();
MatrixXdr enrich_SEjack;
enrich_SEjack=MatrixXdr::Zero(Nbin,1);
 temp_val=0;
for (int i=0;i<Nbin;i++){
    for (int j=0;j<Njack;j++){
        temp_val=enrich_jack(i,j)-sum_row(i);
        temp_val= temp_val* temp_val;
        enrich_SEjack(i,0)+=temp_val;
    }
    enrich_SEjack(i,0)=enrich_SEjack(i,0)*(Njack-1)/Njack;
    enrich_SEjack(i,0)=sqrt(enrich_SEjack(i,0));
}


//for (int i=0;i<Njack;i++)
  //  cout<<jack.col(i).transpose()<<endl;
cout<<"OUTPUT: "<<endl;
for (int j=0;j<Nbin;j++)
     cout<<"h^2 of bin "<<j<<" : "<<point_est(j,0)<<" ,  SE: "<<SEjack(j,0)<<endl;
cout<<"Total h^2 : "<<point_est(Nbin,0)<<" , SE: "<<SEjack(Nbin,0)<<endl;
for (int j=0;j<Nbin;j++)
     cout<<"Enrichment of bin "<<j<<" :"<<enrich_point_est(j,0)<<" ,  SE: "<<enrich_SEjack(j,0)<<endl;

/*cout<<"Point estimates :"<<endl;
cout<<point_est.transpose()<<endl;
cout<<"SEs     :"<<endl;
cout<<SEjack.transpose()<<endl;

cout<<"Enrichment :"<<endl;
cout<<enrich_point_est.transpose()<<endl;
cout<<"SEs     :"<<endl;
cout<<enrich_SEjack.transpose()<<endl;
*/
 std::ofstream outfile;
string add_output=command_line_opts.OUTPUT_FILE_PATH;
outfile.open(add_output.c_str(), std::ios_base::app);

outfile<<"Point estimates :"<<endl;
outfile<<point_est.transpose()<<endl;
outfile<<"SEs     :"<<endl;
outfile<<SEjack.transpose()<<endl;
      
outfile<<"Enrichment :"<<endl;
outfile<<enrich_point_est.transpose()<<endl;
outfile<<"SEs     :"<<endl;
outfile<<enrich_SEjack.transpose()<<endl;








	return 0;
}
