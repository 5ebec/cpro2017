//FC BackWarD
void fc_bwd(int m, int n, const float *x, const float *dy, const float *A, float *dA, float *db, float *dx){
  //dA=dy*x,db=dy,dx=A*dy
  for (int s=0; s<m; s++){
    db[s] = dy[s];
    for (int t=0; t<n; t++){
      dA[s*n+t] = dy[s]*x[t];
    }
  }
  for (int t=0; t<n; t++){
    dx[t] = 0;
    for (int s=0; s<m; s++){
      dx[t] += dA[s*n+t]*dy[s];
    }
  }
}
