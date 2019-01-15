#include <stdio.h>

//Print_octave
void print_oct(int m, int n, const float *x, const char *name){
  printf("%s = [ ", name);
  for(int i=0; i<m*n; i++){
    printf("%.4f", x[i]);
    if((i+1)%n == 0){
      printf(" ;\n");
      if(i != m*n-1){
        printf(" ");
      }
    }else{
      printf("  ");
    }
  }
  printf("];\n");
}

int main(){
  float m[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  print_oct(3, 4, m, "m1");
  return 0;
}
